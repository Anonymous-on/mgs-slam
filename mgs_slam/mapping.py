"""HI-SLAM2 Gaussian mapper driven by MASt3R-SLAM keyframes."""
from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace
import math
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from .vendor import bootstrap
bootstrap()

from gaussian.renderer import render
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.camera_utils import Camera
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.loss_utils import ssim

from .geometry import (
    adaptive_point_mask, infer_intrinsics, pointmap_depth, quaternion_multiply_wxyz,
    sim3_parts, sim3_to_w2c,
)


class GaussianMapper:
    def __init__(self, cfg, intrinsics=None, device='cuda:0'):
        self.cfg = cfg
        self.options = cfg['MGS']
        self.device = device
        self.intrinsics = tuple(intrinsics) if intrinsics is not None else None
        self.model = GaussianModel(sh_degree=int(self.options['sh_degree']), config=cfg)
        self.model.init_lr(float(self.options['scene_extent']))
        self.model.training_setup(SimpleNamespace(**cfg['opt_params']))
        self.background = torch.zeros(3, device=device)
        self.views = OrderedDict()
        self.poses = {}
        self.iteration = 0

    def _camera(self, frame, timestamp):
        depth, valid = pointmap_depth(frame, self.options['min_point_confidence'])
        if self.intrinsics is None:
            self.intrinsics = infer_intrinsics(frame)
        fx, fy, cx, cy = self.intrinsics
        height, width = frame.uimg.shape[:2]
        image = frame.uimg.to(self.device).permute(2, 0, 1).float().contiguous()
        # HI-SLAM2's camera expects the transposed projection matrix.
        projection = getProjectionMatrix2(
            znear=0.01, zfar=float(self.options['far_plane']),
            fx=fx, fy=fy, cx=cx, cy=cy, W=width, H=height,
        ).T.to(self.device)
        normal = torch.zeros_like(image)
        camera = Camera.init_from_tracking(
            image, depth.detach(), normal, sim3_to_w2c(frame.T_WC),
            frame.frame_id, projection, [fx, fy, cx, cy, width, height], timestamp,
        )
        camera.valid_depth = valid
        return camera

    def add_keyframe(self, frame, timestamp):
        camera = self._camera(frame, timestamp)
        idx = frame.frame_id
        if idx in self.views:
            self.views[idx] = camera
            return
        # MASt3R point-map depth and Sim(3) camera use one world scale.
        seed_depth = camera.depth
        if self.options.get('adaptive_sampling', False):
            keep = adaptive_point_mask(
                camera.original_image, seed_depth, camera.valid_depth,
                tile=int(self.options['sample_tile']),
                max_points=int(self.options['sample_max_points']),
            )
            seed_depth = torch.where(keep, seed_depth, torch.zeros_like(seed_depth))
        self.model.extend_from_pcd_seq(
            camera, kf_id=idx, init=(len(self.views) == 0),
            depthmap=seed_depth.detach().cpu().numpy(),
        )
        self.views[idx] = camera
        self.poses[idx] = frame.T_WC.data.detach().clone()
        self.optimize(int(self.options['init_iterations'] if len(self.views) == 1
                          else self.options['mapping_iterations']))

    def apply_pose_updates(self, keyframes):
        """Move Gaussians with their source keyframes after pose-graph corrections."""
        from lietorch import Sim3
        with torch.no_grad():
            for idx, old_data in list(self.poses.items()):
                new_pose = keyframes[idx].T_WC
                if torch.allclose(old_data, new_pose.data, atol=1e-6):
                    continue
                old_pose = Sim3(old_data)
                mask = (self.model.unique_kfIDs == idx).to(self.device)
                if mask.any():
                    local = old_pose.inv().act(self.model._xyz[mask])
                    self.model._xyz[mask] = new_pose.act(local)
                    old_r, _, old_s = sim3_parts(old_pose)
                    new_r, _, new_s = sim3_parts(new_pose)
                    delta = new_r @ old_r.T
                    ratio = (new_s / old_s).clamp_min(1e-8)
                    self.model._scaling[mask] += torch.log(ratio)
                    q_xyzw = Rotation.from_matrix(delta.cpu().numpy()).as_quat()
                    dq = torch.as_tensor(
                        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                        device=self.device, dtype=self.model._rotation.dtype,
                    )
                    self.model._rotation[mask] = F.normalize(
                        quaternion_multiply_wxyz(dq, self.model._rotation[mask]), dim=-1
                    )
                old_data.copy_(new_pose.data)
                if idx in self.views:
                    self.views[idx].update_RT(
                        sim3_to_w2c(new_pose)[:3, :3], sim3_to_w2c(new_pose)[:3, 3]
                    )

    def _loss(self, camera, result):
        predicted_rgb = result['render']
        predicted_depth = result['depth'].squeeze(0)
        target_rgb = camera.original_image
        target_depth = camera.depth.to(self.device)
        valid = camera.valid_depth & torch.isfinite(predicted_depth) & (predicted_depth > 1e-5)
        rgb_l1 = F.l1_loss(predicted_rgb, target_rgb)
        rgb_ssim = 1.0 - ssim(predicted_rgb[None], target_rgb[None])
        if valid.any():
            inverse_depth = F.l1_loss(
                predicted_depth[valid].clamp_min(1e-5).reciprocal(),
                target_depth[valid].clamp_min(1e-5).reciprocal(),
            )
        else:
            inverse_depth = rgb_l1.new_zeros(())
        dssim = float(self.cfg['opt_params']['lambda_dssim'])
        return ((1-dssim)*rgb_l1 + dssim*rgb_ssim
                + float(self.options['depth_loss_weight'])*inverse_depth)

    def optimize(self, iterations):
        if not self.views or self.model.get_xyz.numel() == 0:
            return
        window_size = int(self.cfg['Training']['window_size'])
        windows = list(self.views.values())[-window_size:]
        for step in range(iterations):
            self.iteration += 1
            camera = windows[step % len(windows)]
            self.model.update_learning_rate(self.iteration)
            result = render(camera, self.model, self.background)
            loss = self._loss(camera, result)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite Gaussian mapping loss at {self.iteration}')
            loss.backward()
            with torch.no_grad():
                visible = result['visibility_filter']
                self.model.max_radii2D[visible] = torch.maximum(
                    self.model.max_radii2D[visible], result['radii'][visible]
                )
                self.model.add_densification_stats(result['viewspace_points'], visible)
                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)
                period = int(self.options['densify_every'])
                if period > 0 and self.iteration % period == 0:
                    self.model.densify_and_prune(
                        float(self.cfg['opt_params']['densify_grad_threshold']),
                        float(self.options['min_opacity']),
                        float(self.options['scene_extent']),
                        int(self.cfg['Training']['size_threshold']),
                    )

    def save(self, path):
        self.model.save_ply(str(path))

    @torch.no_grad()
    def render_keyframe(self, frame_id):
        return render(self.views[frame_id], self.model, self.background)['render']

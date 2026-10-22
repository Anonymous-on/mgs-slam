"""Coordinate and point-map adapters between MASt3R-SLAM and HI-SLAM2."""
from __future__ import annotations

import torch


def sim3_parts(pose):
    """Return R, t, positive scale from a camera-to-world lietorch Sim3."""
    matrix = pose.matrix().reshape(4, 4)
    scaled_rotation = matrix[:3, :3]
    scale = torch.linalg.det(scaled_rotation).abs().clamp_min(1e-12).pow(1.0 / 3.0)
    rotation = scaled_rotation / scale
    return rotation, matrix[:3, 3], scale


def sim3_to_w2c(pose):
    rotation, translation, _ = sim3_parts(pose)
    w2c = torch.eye(4, dtype=rotation.dtype, device=rotation.device)
    w2c[:3, :3] = rotation.T
    w2c[:3, 3] = -rotation.T @ translation
    return w2c


def pointmap_depth(frame, min_confidence=0.0):
    """Convert the camera-local point map into rasterizer depth units.

    Sim(3) scale is applied because the Gaussian world points are transformed by
    the same Sim(3), whereas the renderer accepts only a rigid w2c camera.
    """
    height, width = frame.uimg.shape[:2]
    points = frame.X_canon.reshape(height, width, 3)
    confidence = frame.get_average_conf().reshape(height, width)
    _, _, scale = sim3_parts(frame.T_WC)
    depth = points[..., 2] * scale
    valid = torch.isfinite(points).all(-1) & torch.isfinite(confidence)
    valid &= (points[..., 2] > 1e-5) & (confidence > min_confidence)
    return torch.where(valid, depth, torch.zeros_like(depth)), valid


def infer_intrinsics(frame, max_samples=50000):
    """Fit a pinhole K from a MASt3R point map when no calibration is supplied."""
    height, width = frame.uimg.shape[:2]
    points = frame.X_canon.reshape(-1, 3)
    z = points[:, 2]
    valid = torch.isfinite(points).all(-1) & (z > 1e-5)
    ids = torch.where(valid)[0]
    if len(ids) < 32:
        raise ValueError('Not enough valid point-map pixels to estimate intrinsics')
    if len(ids) > max_samples:
        ids = ids[torch.linspace(0, len(ids) - 1, max_samples, device=ids.device).long()]
    x = points[ids, 0] / z[ids]
    y = points[ids, 1] / z[ids]
    u = (ids % width).float() + 0.5
    v = torch.div(ids, width, rounding_mode='floor').float() + 0.5
    au = torch.stack((x, torch.ones_like(x)), -1)
    av = torch.stack((y, torch.ones_like(y)), -1)
    fx, cx = torch.linalg.lstsq(au, u[:, None]).solution[:, 0]
    fy, cy = torch.linalg.lstsq(av, v[:, None]).solution[:, 0]
    if not bool(torch.isfinite(torch.stack((fx, fy, cx, cy))).all()):
        raise ValueError('Point-map intrinsics fit produced non-finite values')
    if fx <= 0 or fy <= 0:
        raise ValueError('Point-map intrinsics fit produced non-positive focal lengths')
    return (float(fx), float(fy), float(cx), float(cy))


def quaternion_multiply_wxyz(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ), -1)


def adaptive_point_mask(rgb, depth, valid, tile=16, max_points=15000):
    """Geometry/texture-aware point selection for Gaussian initialization.

    More valid points are retained near image and depth changes than in flat
    regions; one representative is kept per tile, plus strong edge pixels.
    """
    import torch.nn.functional as F
    height, width = depth.shape
    gray = rgb.float().mean(0)
    gx = F.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1))
    gy = F.pad((gray[1:] - gray[:-1]).abs(), (0, 0, 0, 1))
    dx = F.pad((depth[:, 1:] - depth[:, :-1]).abs(), (0, 1))
    dy = F.pad((depth[1:] - depth[:-1]).abs(), (0, 0, 0, 1))
    score = gx + gy + 0.1 * (dx + dy) / depth.clamp_min(1e-3)
    score = torch.where(valid, score, torch.full_like(score, -1.0))
    selected = torch.zeros_like(valid)
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            patch = score[y:y+tile, x:x+tile]
            if patch.max() < 0:
                continue
            offset = int(torch.argmax(patch))
            py, px = divmod(offset, patch.shape[1])
            selected[y+py, x+px] = True
    # Preserve extra high-frequency image/depth edges up to a bounded budget.
    remaining = max(0, max_points - int(selected.sum()))
    if remaining:
        candidates = score.masked_fill(selected, -1.0).flatten()
        take = min(remaining, int((candidates >= 0).sum()))
        if take:
            ids = torch.topk(candidates, k=take).indices
            selected.flatten()[ids] = True
    return selected & valid

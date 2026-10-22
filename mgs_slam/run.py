"""Monocular MASt3R-SLAM tracking plus HI-SLAM2 Gaussian mapping."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import lietorch
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from .vendor import bootstrap
bootstrap()

from mast3r_slam.config import config as tracking_config, load_config
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.frame import SharedKeyframes, create_frame
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
from mast3r_slam.tracker import FrameTracker

from .geometry import sim3_parts
from .mapping import GaussianMapper


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='Image directory, video, or MASt3R-SLAM dataset')
    p.add_argument('--mast3r-checkpoint', required=True)
    p.add_argument('--retrieval-checkpoint', default=None)
    p.add_argument('--tracking-config', default='config/mast3r.yaml')
    p.add_argument('--mapping-config', default='config/mgs.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--intrinsics', nargs=4, type=float, metavar=('FX','FY','CX','CY'))
    return p.parse_args()


def save_trajectory(path, entries, keyframes):
    with open(path, 'w') as out:
        out.write('# timestamp tx ty tz qx qy qz qw (camera-to-world)\n')
        for timestamp, anchor_index, relative in entries:
            pose = keyframes[anchor_index].T_WC * relative
            rotation, translation, _ = sim3_parts(pose)
            q = Rotation.from_matrix(rotation.detach().cpu().numpy()).as_quat()
            t = translation.detach().cpu().numpy()
            out.write(' '.join(map(str, (timestamp, *t, *q))) + '\n')


def run(args):
    load_config(args.tracking_config)
    with open(args.mapping_config) as stream:
        mapping_config = yaml.safe_load(stream)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / 'run_config.json', 'w') as stream:
        json.dump(vars(args), stream, indent=2)
    dataset = load_dataset(args.dataset)
    if args.stride > 1:
        dataset.subsample(args.stride)
    if len(dataset) == 0:
        raise ValueError(f'No images found in {args.dataset}')
    height, width = dataset.get_img_shape()[0]
    manager = mp.Manager()
    keyframes = SharedKeyframes(manager, height, width,
                                buffer=int(mapping_config['MGS']['max_keyframes']),
                                device=args.device)
    model = load_mast3r(args.mast3r_checkpoint, args.device).eval()
    tracker = FrameTracker(model, keyframes, args.device)
    graph = FactorGraph(model, keyframes, None, args.device)
    retriever = (load_retriever(model, args.retrieval_checkpoint, args.device)
                 if args.retrieval_checkpoint else None)
    mapper = GaussianMapper(mapping_config, args.intrinsics, args.device)
    trajectory = []
    events = []
    limit = min(len(dataset), args.max_frames) if args.max_frames else len(dataset)
    begin = time.monotonic()
    last_pose = lietorch.Sim3.Identity(1, device=args.device)
    try:
        for index in range(limit):
            timestamp, image = dataset[index]
            initial = last_pose
            frame = create_frame(index, image, initial, dataset.img_size, args.device)
            if index == 0:
                points, confidence = mast3r_inference_mono(model, frame)
                frame.update_pointmap(points, confidence)
                keyframes.append(frame)
                mapper.add_keyframe(keyframes[0], timestamp)
                status = 'initialized'
            else:
                is_keyframe, _, needs_reloc = tracker.track(frame)
                if needs_reloc:
                    events.append({'frame': index, 'status': 'tracking_failed'})
                    print(f'[MGS] frame={index}: tracking failed, skipped')
                    continue
                status = 'tracked'
                if is_keyframe:
                    if len(keyframes) >= keyframes.buffer:
                        raise RuntimeError('Keyframe buffer full; increase MGS.max_keyframes')
                    keyframes.append(frame)
                    newest = len(keyframes) - 1
                    candidates = [newest - 1]
                    if retriever is not None:
                        found = retriever.update(
                            frame, add_after_query=True,
                            k=tracking_config['retrieval']['k'],
                            min_thresh=tracking_config['retrieval']['min_thresh'],
                        )
                        candidates.extend(j for j in found if j < newest - 1)
                    candidates = sorted(set(candidates))
                    graph.add_factors(
                        candidates, [newest] * len(candidates),
                        tracking_config['local_opt']['min_match_frac'],
                    )
                    if graph.ii.numel() > 0:
                        graph.solve_GN_rays()
                    mapper.apply_pose_updates(keyframes)
                    frame = keyframes[newest]
                    mapper.add_keyframe(frame, timestamp)
                    tracker.reset_idx_f2k()
                    status = 'keyframe'
            last_pose = lietorch.Sim3(frame.T_WC.data.detach().clone())
            anchor_index = len(keyframes) - 1
            relative = keyframes[anchor_index].T_WC.inv() * last_pose
            trajectory.append((timestamp, anchor_index,
                               lietorch.Sim3(relative.data.detach().clone())))
            events.append({'frame': index, 'timestamp': float(timestamp),
                           'status': status, 'keyframes': len(keyframes),
                           'gaussians': int(mapper.model.get_xyz.shape[0])})
            print(f'[MGS] frame={index}/{limit} {status}, keyframes={len(keyframes)}, '
                  f'gaussians={mapper.model.get_xyz.shape[0]}', flush=True)
        mapper.optimize(int(mapping_config['MGS']['final_iterations']))
        mapper.save(output / 'gaussians.ply')
        save_trajectory(output / 'trajectory.txt', trajectory, keyframes)
        with open(output / 'events.json', 'w') as stream:
            json.dump(events, stream, indent=2)
        print(f'[MGS] done in {time.monotonic()-begin:.1f}s: {output}')
    finally:
        manager.shutdown()


def main():
    mp.set_start_method('spawn', force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    run(parse_args())


if __name__ == '__main__':
    main()

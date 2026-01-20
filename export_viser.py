#!/usr/bin/env python3
"""
Export viser_cache.pkl to .viser format for static web playback.

Usage:
    python export_viser.py --input /path/to/decay_0.0/viser_cache.pkl --output assets/viser/scene.viser
    python export_viser.py --input-dir /path/to/decay_outputs_streaming/uuid --output-dir assets/viser/
"""

import argparse
import pickle
import os
import glob
from pathlib import Path

import numpy as np
import viser


def export_single_pkl(pkl_path: str, output_path: str, fps: float = 30.0, point_size: float = 0.007):
    """
    Export a single viser_cache.pkl to .viser format.

    Args:
        pkl_path: Path to viser_cache.pkl
        output_path: Output .viser file path
        fps: Playback frame rate
        point_size: Point cloud point size
    """
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        frames = pickle.load(f)

    if not frames:
        print(f"  Warning: No frames in {pkl_path}, skipping.")
        return False

    print(f"  Found {len(frames)} frames")

    # Create viser server (headless, won't actually serve)
    server = viser.ViserServer(port=None)  # port=None for headless

    # Set up direction (matching server_streaming.py)
    server.scene.set_up_direction("-y")

    # Get serializer
    serializer = server.get_scene_serializer()

    # Process each frame
    frame_duration = 1.0 / fps

    for i, frame in enumerate(frames):
        # Clear previous frame's objects by using the same names (they get overwritten)

        # Add point cloud
        pts = frame["points"]
        clrs = frame["colors"]

        # Ensure colors are in correct format (0-255 uint8 or 0-1 float)
        if clrs.max() <= 1.0:
            clrs_uint8 = (clrs * 255).astype(np.uint8)
        else:
            clrs_uint8 = clrs.astype(np.uint8)

        server.scene.add_point_cloud(
            name="/video",
            points=pts.astype(np.float32),
            colors=clrs_uint8,
            point_size=point_size,
        )

        # Add camera frustum
        server.scene.add_camera_frustum(
            name="/camera",
            fov=1.0,
            aspect=frame["aspect"],
            scale=0.05,
            position=tuple(frame["cam_pos"]),
            wxyz=tuple(frame["cam_quat"]),
            image=frame["image"].astype(np.uint8) if frame["image"] is not None else None,
        )

        # Insert delay for animation timing
        if i < len(frames) - 1:
            serializer.insert_sleep(frame_duration)

    # Serialize and save
    print(f"  Serializing...")
    data = serializer.serialize()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Path(output_path).write_bytes(data)

    print(f"  Saved to {output_path} ({len(data) / 1024 / 1024:.2f} MB)")

    server.stop()
    return True


def export_project_dir(input_dir: str, output_dir: str, fps: float = 30.0):
    """
    Export all decay levels from a project directory.

    Args:
        input_dir: Path to project directory (e.g., decay_outputs_streaming/uuid)
        output_dir: Output directory for .viser files
    """
    # Find all decay subdirectories
    decay_dirs = sorted(glob.glob(os.path.join(input_dir, "decay_*")))

    if not decay_dirs:
        print(f"No decay_* directories found in {input_dir}")
        return

    print(f"Found {len(decay_dirs)} decay levels")

    os.makedirs(output_dir, exist_ok=True)

    for decay_dir in decay_dirs:
        pkl_path = os.path.join(decay_dir, "viser_cache.pkl")
        if not os.path.exists(pkl_path):
            print(f"  Skipping {decay_dir}: no viser_cache.pkl")
            continue

        # Extract decay value from directory name (e.g., "decay_0.5" -> "0.5")
        decay_name = os.path.basename(decay_dir)
        output_path = os.path.join(output_dir, f"{decay_name}.viser")

        export_single_pkl(pkl_path, output_path, fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Export viser_cache.pkl to .viser format")
    parser.add_argument("--input", "-i", type=str, help="Input viser_cache.pkl file")
    parser.add_argument("--output", "-o", type=str, help="Output .viser file")
    parser.add_argument("--input-dir", type=str, help="Input project directory (exports all decay levels)")
    parser.add_argument("--output-dir", type=str, default="assets/viser", help="Output directory for .viser files")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback frame rate")
    parser.add_argument("--point-size", type=float, default=0.007, help="Point cloud point size")

    args = parser.parse_args()

    if args.input:
        # Single file export
        output = args.output or args.input.replace(".pkl", ".viser")
        export_single_pkl(args.input, output, fps=args.fps, point_size=args.point_size)
    elif args.input_dir:
        # Directory export
        export_project_dir(args.input_dir, args.output_dir, fps=args.fps)
    else:
        parser.print_help()
        print("\nError: Must specify --input or --input-dir")


if __name__ == "__main__":
    main()

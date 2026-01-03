"""
Utilities for reading COLMAP sparse reconstructions (cameras / poses).

We rely on COLMAP's official `read_write_model.py` helper which is vendored
inside the repository under `colmap/scripts/python`. If that helper cannot be
imported (e.g. the checkout is missing), the parser gracefully returns an empty
dictionary so downstream callers can fall back to single-view behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def _import_colmap_reader() -> Optional[object]:
    """
    Attempt to import COLMAP's Python helper for reading binary models.
    """
    colmap_scripts = Path(__file__).resolve().parents[2] / "colmap" / "scripts" / "python"
    if not colmap_scripts.exists():
        return None

    added_to_path = False
    if str(colmap_scripts) not in sys.path:
        sys.path.insert(0, str(colmap_scripts))
        added_to_path = True

    try:
        import read_write_model as colmap_rw  # type: ignore
    except ImportError:
        colmap_rw = None
    finally:
        if added_to_path:
            # remove the path we inserted to keep sys.path tidy
            sys.path.pop(0)

    return colmap_rw


def _camera_intrinsics(camera) -> np.ndarray:
    """
    Build a simple 3x3 intrinsic matrix from a COLMAP camera specification.
    The mapping handles the common pinhole / simple radial models explicitly
    and falls back to the first parameters for any other model types.
    """
    K = np.eye(3, dtype=np.float32)
    params = camera.params
    model = getattr(camera, "model", "").upper()

    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        fx = params[0] if len(params) > 0 else camera.width / 2.0
        fy = params[1] if len(params) > 1 else fx
        cx = params[2] if len(params) > 2 else camera.width / 2.0
        cy = params[3] if len(params) > 3 else camera.height / 2.0

    K[0, 0] = float(fx)
    K[1, 1] = float(fy)
    K[0, 2] = float(cx)
    K[1, 2] = float(cy)
    return K


def parse_sparse_model(sparse_dir: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Loads COLMAP sparse reconstruction from `sparse_dir` and returns per-image metadata.

    Returns:
        dict mapping the raw image filename (e.g. `001.jpg`) to tensors:
            {
                "intrinsics": torch.Tensor [3, 3],
                "world_to_cam": torch.Tensor [4, 4],
                "cam_to_world": torch.Tensor [4, 4],
                "resolution": torch.Tensor [2],  # (height, width)
            }
    """
    colmap_rw = _import_colmap_reader()
    if colmap_rw is None:
        return {}

    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    if not cameras_path.exists() or not images_path.exists():
        return {}

    cameras = colmap_rw.read_cameras_binary(str(cameras_path))
    images = colmap_rw.read_images_binary(str(images_path))

    metadata: Dict[str, Dict[str, torch.Tensor]] = {}
    for image in images.values():
        camera = cameras.get(image.camera_id)
        if camera is None:
            continue

        R = image.qvec2rotmat().astype(np.float32)
        t = image.tvec.astype(np.float32)

        world_to_cam = np.eye(4, dtype=np.float32)
        world_to_cam[:3, :3] = R
        world_to_cam[:3, 3] = t

        cam_to_world = np.eye(4, dtype=np.float32)
        cam_to_world[:3, :3] = R.T
        cam_to_world[:3, 3] = -R.T @ t

        K = _camera_intrinsics(camera)

        metadata[Path(image.name).name] = {
            "intrinsics": torch.from_numpy(K),
            "world_to_cam": torch.from_numpy(world_to_cam),
            "cam_to_world": torch.from_numpy(cam_to_world),
            "resolution": torch.tensor([camera.height, camera.width], dtype=torch.float32),
        }

    return metadata


from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms

from hbird.utils.colmap_utils import parse_sparse_model


def sample_uniform_views(candidates: List[Path], num_samples: int) -> List[Path]:
    """
    Uniformly sample `num_samples` paths from `candidates`. When there are fewer
    candidates than samples, repeat the last entry.
    """
    if num_samples <= 0:
        return []

    deduped = sorted({path for path in candidates if path.exists()})
    if not deduped:
        return []

    if len(deduped) >= num_samples:
        if num_samples == 1:
            return [deduped[0]]
        step = (len(deduped) - 1) / float(num_samples - 1)
        indices = [int(round(i * step)) for i in range(num_samples)]
        return [deduped[idx] for idx in indices]

    output = deduped.copy()
    while len(output) < num_samples:
        output.append(output[-1])
    return output[:num_samples]


class MVImgNetDataModule(pl.LightningDataModule):
    """
    LightningDataModule for MVImgNet.

    Loads training and validation datasets from folders structured as:
        <class_id>/<angle_bin>/{img, mask}/<filename>

    Optionally returns segmentation masks (binarized + scaled by class index).
    """

    # Manually defined expected classes
    CLASS_IDX_TO_NAME = [  # background + 15 classes
        'background', 'stove', 'sofa', 'microwave', 'bed', 'toy_cat', 'toy_cow',
        'toy_dragon', 'coat_rack', 'guitar_stand', 'ceiling_lamp', 'toilet',
        'sink', 'strings', 'broccoli', 'durian'
    ]

    def __init__(
        self,
        data_dir: str,
        train_bins: List[str],
        val_bins: List[str],
        train_transforms: Callable,
        val_transforms: Callable,
        batch_size: int,
        num_workers: int,
        return_masks: bool = True,  # ToDo: the default is false for other datasets
        shuffle: bool = False,
        drop_last: bool = True,
        sequence_length: int = 1,
        raw_data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.raw_data_dir = (
            Path(raw_data_dir)
            if raw_data_dir is not None
            else self._infer_raw_dataset_root(self.data_dir)
        )
        self.train_bins = train_bins
        self.val_bins = val_bins
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_masks = return_masks
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sequence_length = sequence_length

        self.train_dataset = None
        self.val_dataset = None
        self.object_to_views: Optional[Dict[str, List[Path]]] = None
        self.object_camera_meta: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

        # Manually defined expected classes
        self.classes = [7, 8, 19, 46, 57, 60, 70, 99, 100, 113, 125, 126, 152, 166, 196]  # 15 classes

        # Validate that dataset structure matches the expected classes
        class_dirs = [p for p in self.data_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        found_class_ids = sorted([int(p.name) for p in class_dirs])

        if found_class_ids != self.classes:
            raise ValueError(
                f"Class mismatch:\n"
                f"  Expected: {self.classes}\n"
                f"  Found:    {found_class_ids}\n\n"
                f"Please verify that the dataset folder structure matches the class list."
            )

        # Build class-to-index mapping from validated class list
        self.class_to_index = {str(class_id): idx + 1 for idx, class_id in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.train_dataset) if self.train_dataset else 0

    def get_train_dataset_size(self) -> int:
        return len(self.train_dataset)

    def get_val_dataset_size(self) -> int:
        return len(self.val_dataset)

    def get_num_classes(self) -> int:
        return len(self.CLASS_IDX_TO_NAME)
    
    def class_id_to_name(self, idx: int) -> str:
        return self.CLASS_IDX_TO_NAME[idx]
    
    def setup(self, stage: Optional[str] = None):
        
        # Construct "training" dataset only if train_bins is provided
        if self.object_to_views is None:
            self.object_to_views = self._build_object_to_views()
        if self.object_camera_meta is None:
            self.object_camera_meta = self._load_camera_metadata()

        if self.train_bins is not None:
            train_bin_paths = [
                self.data_dir / str(class_id) / str(bin)
                for class_id in self.classes
                for bin in self.train_bins
                if (self.data_dir / str(class_id) / str(bin)).exists()
            ]

            self.train_dataset = MVImgNetDataset(
                bin_paths=train_bin_paths,
                transforms=self.train_transforms,
                return_masks=self.return_masks,
                class_to_index=self.class_to_index,
                sequence_length=self.sequence_length,
                object_to_views=self.object_to_views,
                camera_metadata=self.object_camera_meta,
            )
        else:
            self.train_dataset = []

        # Construct "validation" dataset only if val_bins is provided
        if self.val_bins is not None:
            val_bin_paths = [
                self.data_dir / str(class_id) / str(bin)
                for class_id in self.classes
                for bin in self.val_bins
                if (self.data_dir / str(class_id) / str(bin)).exists()
            ]

            self.val_dataset = MVImgNetDataset(
                bin_paths=val_bin_paths,
                transforms=self.val_transforms,
                return_masks=self.return_masks,
                class_to_index=self.class_to_index,
                sequence_length=self.sequence_length,
                object_to_views=self.object_to_views,
                camera_metadata=self.object_camera_meta,
            )
        else:
            self.val_dataset = []

        print(f"✅ MVImgNet Loaded → Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True,
        )

    @staticmethod
    def _infer_raw_dataset_root(angle_dir: Path) -> Path:
        """
        Attempt to infer the raw MVImgNet root given the split-angle directory.
        """
        candidate = angle_dir.parent / "mvimgnet"
        return candidate

    def _build_object_to_views(self) -> Dict[str, List[Path]]:
        """
        Build a mapping from object IDs to all available raw MVImgNet image paths.
        """
        mapping: Dict[str, List[Path]] = defaultdict(list)
        if not self.raw_data_dir.exists():
            print(f"⚠️ Raw MVImgNet directory not found: {self.raw_data_dir}")
            return mapping

        for class_dir in self.raw_data_dir.iterdir():
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            for object_dir in class_dir.iterdir():
                if not object_dir.is_dir():
                    continue
                images_dir = object_dir / "images"
                if not images_dir.exists():
                    continue
                object_id = object_dir.name
                for img_path in sorted(images_dir.glob("*.jpg")):
                    mapping[object_id].append(img_path)
        return mapping

    def _load_camera_metadata(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        metadata: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for class_dir in self.raw_data_dir.iterdir():
            if not class_dir.is_dir() or not class_dir.name.isdigit():
                continue
            for object_dir in class_dir.iterdir():
                if not object_dir.is_dir():
                    continue
                sparse_dir = object_dir / "sparse" / "0"
                if not sparse_dir.exists():
                    continue
                object_id = object_dir.name
                metadata[object_id] = parse_sparse_model(sparse_dir)
        return metadata
    

class MVImgNetDataset(Dataset):
    """
    PyTorch Dataset for MVImgNet, a multi-view image dataset with class-specific folder structure.

    Expected directory structure:
        <class_id>/<angle_bin>/{img, mask}/<filename>
        e.g., 7/15/img/cat.jpg and 7/15/mask/cat.jpg.png

    The class label is inferred from the directory name three levels above each image file.
    Each unique class ID is mapped to a sequential integer index starting from 1.

    Masks are optionally returned. They are binarized and scaled by the corresponding class index.

    Args:
        bin_paths (List[Path]): List of angle bin folders (e.g., [.../7/15, .../19/30]).
        transforms (Optional[Callable]): Image/mask transforms to apply.
        return_masks (bool): Whether to load and return segmentation masks.
        class_to_index (Dict[str, int]): Mapping from original class IDs to sequential indices.

    Returns:
        Tuple[Image, Tensor]: If return_masks is True.
        Image: If return_masks is False.
    """

    def __init__(
        self,
        bin_paths: List[Path],  # List of angle bin folders like ["folder/path/1", "folder/path/2", ...]
        transforms: Optional[Callable] = None,
        return_masks: bool = True,  # ToDo: the default is false for other classes
        class_to_index: Dict[str, int] = None,
        sequence_length: int = 1,
        object_to_views: Optional[Dict[str, List[Path]]] = None,
        camera_metadata: Optional[Dict[str, Dict[str, Dict[str, torch.Tensor]]]] = None,
    ):
        self.bin_paths = [Path(p) for p in bin_paths]
        self.transforms = transforms
        self.return_masks = return_masks
        self.class_to_index = class_to_index
        self.sequence_length = max(sequence_length, 1)
        self.object_to_views = object_to_views if object_to_views is not None else {}
        self.camera_metadata = camera_metadata if camera_metadata is not None else {}
        self.samples = self._collect()

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        query_path: Path = sample["query_image"]
        mask_path: Optional[Path] = sample["mask_path"]
        class_id: int = sample["class_id"]
        angle_bin: str = sample["bin"]
        object_id: str = sample["object_id"]
        support_candidates: List[Path] = sample["support_paths"]
        raw_query_path: Optional[Path] = sample.get("raw_query_path")
        camera_info = self.camera_metadata.get(object_id, {})

        views: List[torch.Tensor] = []
        mask_tensor: Optional[torch.Tensor] = None
        intrinsics_list: List[torch.Tensor] = []
        world_to_cam_list: List[torch.Tensor] = []
        cam_to_world_list: List[torch.Tensor] = []

        query_image = Image.open(query_path).convert("RGB")
        query_mask = Image.open(mask_path) if (mask_path is not None and mask_path.exists()) else None
        query_tensor, mask_tensor = self._apply_transforms(query_image, query_mask, class_id)
        views.append(query_tensor)

        def append_camera_meta(meta: Optional[Dict[str, torch.Tensor]]):
            if meta is None:
                intrinsics_list.append(torch.eye(3, dtype=torch.float32))
                world_to_cam_list.append(torch.eye(4, dtype=torch.float32))
                cam_to_world_list.append(torch.eye(4, dtype=torch.float32))
            else:
                intrinsics_list.append(meta["intrinsics"].float())
                world_to_cam_list.append(meta["world_to_cam"].float())
                cam_to_world_list.append(meta["cam_to_world"].float())

        query_meta = None
        if raw_query_path is not None:
            query_meta = camera_info.get(Path(raw_query_path).name)
        append_camera_meta(query_meta)

        support_needed = max(self.sequence_length - 1, 0)
        selected_supports = sample_uniform_views(support_candidates, support_needed)
        for support_path in selected_supports:
            try:
                support_img = Image.open(support_path).convert("RGB")
            except (FileNotFoundError, OSError):
                support_img = Image.open(query_path).convert("RGB")
            support_tensor, _ = self._apply_transforms(support_img, None, class_id)
            views.append(support_tensor)

            support_meta = camera_info.get(support_path.name)
            append_camera_meta(support_meta)

        while len(views) < self.sequence_length:
            views.append(views[-1].clone())
            append_camera_meta(None)

        views_tensor = torch.stack(views, dim=0)
        intrinsics_tensor = torch.stack(intrinsics_list, dim=0)
        world_to_cam_tensor = torch.stack(world_to_cam_list, dim=0)
        cam_to_world_tensor = torch.stack(cam_to_world_list, dim=0)
        example: Dict[str, Any] = {
            "views": views_tensor,
            "label": torch.tensor(class_id, dtype=torch.long),
            "bin": angle_bin,
            "object_id": object_id,
            "camera": {
                "intrinsics": intrinsics_tensor,
                "world_to_cam": world_to_cam_tensor,
                "cam_to_world": cam_to_world_tensor,
            },
        }
        if mask_tensor is not None:
            example["mask"] = mask_tensor
        return example
    
    # Internal methods:

    def _get_class_index(self, path_to_img: Path) -> int:
        """
        Retrieves the class index for a given image path using the provided class-to-index mapping.

        The class index is determined from the name of the folder three levels above
        the image path (i.e., the original class ID as a string).

        Args:
            path_to_img (Path): Path to an image file.

        Returns:
            int: The class index associated with the original class ID.
        """
        # The original class ID is taken from the directory name three levels above the image
        original_class_id = path_to_img.parent.parent.parent.name  # e.g. "70"
        try:
            return self.class_to_index[original_class_id]  # 1 … 15
        except KeyError:
            raise KeyError(f"Class ID '{original_class_id}' not found in class_to_index mapping.")
    
    def _collect(self) -> List[Dict[str, Any]]:
        """
        Collect metadata for each query frame defined in the angle-bin split.
        """
        samples: List[Dict[str, Any]] = []
        for bin_path in self.bin_paths:
            angle_bin = bin_path.name
            class_id_str = bin_path.parent.name
            img_dir = bin_path / "img"
            mask_dir = bin_path / "mask"

            if not img_dir.exists():
                print(f"⚠️ Missing image dir: {img_dir}")
                continue
            if self.class_to_index is None or class_id_str not in self.class_to_index:
                print(f"⚠️ Skipping unknown class ID: {class_id_str}")
                continue

            for img_file in sorted(img_dir.glob("*.jpg")):
                mask_file = mask_dir / f"{img_file.name}.png"
                if self.return_masks and not mask_file.is_file():
                    print(f"⚠️ Skipping due to missing mask: {mask_file}")
                    continue

                object_id = img_file.stem.split("_")[0]
                support_views = sorted(self.object_to_views.get(object_id, []))

                raw_query = self._resolve_raw_view(object_id, img_file.name)

                samples.append(
                    {
                        "query_image": img_file,
                        "mask_path": mask_file if self.return_masks else None,
                        "class_id": self.class_to_index[class_id_str],
                        "bin": angle_bin,
                        "object_id": object_id,
                        "support_paths": support_views,
                        "raw_query_path": raw_query,
                    }
                )
        return samples

    def _resolve_raw_view(self, object_id: str, formatted_name: str) -> Optional[Path]:
        """
        Matches an angle-binned filename like <object_id>_001.jpg to the raw images/<frame>.jpg path.
        """
        candidates = self.object_to_views.get(object_id, [])
        if not candidates:
            return None
        suffix = formatted_name.split("_")[-1]
        for candidate in candidates:
            if candidate.name == suffix:
                return candidate
        # fall back to first candidate
        return candidates[0]

    def _apply_transforms(
        self,
        image: Image.Image,
        mask: Optional[Image.Image],
        class_id: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_out = image
        mask_out = mask if mask is not None else Image.new("L", image.size, color=0)

        if self.transforms is not None:
            img_out, mask_out = self.transforms(img_out, mask_out)
        else:
            img_out = transforms.ToTensor()(img_out)
            mask_out = torch.from_numpy(np.array(mask_out, dtype=np.float32))

        mask_tensor: Optional[torch.Tensor] = None
        if mask is not None:
            if isinstance(mask_out, torch.Tensor):
                mask_tensor = mask_out.clone().float()
            else:
                mask_array = np.array(mask_out, dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array)
            mask_tensor = (mask_tensor > 0).float() * float(class_id)
            mask_tensor = mask_tensor.unsqueeze(0)

        return img_out, mask_tensor
    
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


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
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_bins = train_bins
        self.val_bins = val_bins
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_masks = return_masks
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.train_dataset = None
        self.val_dataset = None

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
    ):
        self.bin_paths = [Path(p) for p in bin_paths]
        self.transforms = transforms
        self.return_masks = return_masks
        self.class_to_index = class_to_index
        self.images, self.masks = self._collect()

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]) if self.return_masks else None  # values in [0,255]

        if self.transforms:
            if self.return_masks:
                img, mask = self.transforms(img, mask)

                # Convert mask to binary so that values are in {0, 1}.
                # The masks contain values between 0 and 255, but we are not interested
                # in "0.3 object", so we consider an object everything that is above 0.
                # The threshold is chosen based on dataset inspection (see mvimgnet_masks_vs_preds.ipynb).
                mask = (mask > 0).float() 

                # Convert mask to multi-class format so that:
                # background = 0; object belongs to (0,1) depending on the class index
                # It is important that the object is not 1.
                path_to_img = self.images[index]
                class_index = self._get_class_index(path_to_img)  # in [0, N-1]
                # Division of the mask by 255 is done to avoid mask values of an object to become 1.
                # See the create_memory() function for more details.
                mask = mask * class_index / 255.0

            else:
                img = self.transforms(img)

        return (img, mask) if mask is not None else img
    
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
    
    def _collect(self) -> Tuple[List[Path], List[Optional[Path]]]:
        """
        Collects and validates image and mask paths from the given bin directories.

        Expects images to be in 'img/' and masks in 'mask/' subdirectories. Masks are expected
        to be named as '{image_name}.png' (e.g., 'cat.jpg.png').

        Returns:
            Tuple[List[Path], List[Optional[Path]]]: Lists of image and corresponding mask paths.
        """
        image_paths = []
        mask_paths = []

        for bin_path in self.bin_paths:
            img_dir = bin_path / "img"
            mask_dir = bin_path / "mask"

            if not img_dir.exists():
                print(f"⚠️ Missing image dir: {img_dir}")
                continue

            for img_file in sorted(img_dir.glob("*.jpg")):
                mask_file = mask_dir / f"{img_file.name}.png"  # note that the masks are named as e.g.: cat.jpg.png
                if self.return_masks and not mask_file.is_file():
                    print(f"⚠️ Skipping due to missing mask: {mask_file}")
                    continue

                image_paths.append(img_file)
                mask_paths.append(mask_file if self.return_masks else None)
        
        assert all(p.is_file() for p in image_paths)
        if self.return_masks:
            assert all(p.is_file() for p in mask_paths)

        return image_paths, mask_paths
    
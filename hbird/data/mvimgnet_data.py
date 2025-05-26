from pathlib import Path
from typing import Optional, Callable, Tuple, Any, List
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class MVImgNetDataModule(pl.LightningDataModule):
    CLASS_IDX_TO_NAME = ['background', 'stove', 'sofa', 'microwave', 'bed', 'toy_cat', 'toy_cow', 'toy_dragon', 'coat_rack', 'guitar_stand', 'ceiling_lamp', 'toilet', 'sink', 'strings', 'broccoli', 'durian']

    def __init__(
        self,
        data_dir: str,
        train_bins: List[str],
        val_bins: List[str],
        train_transforms: Callable,
        val_transforms: Callable,
        batch_size: int,
        num_workers: int,
        return_masks: bool = True,
        shuffle: bool = False,
        drop_last: bool = True,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_bins = train_bins
        self.val_bins = val_bins
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_masks = return_masks
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.classes = [7, 8, 19, 46, 57, 60, 70, 99, 100, 113, 125, 126, 152, 166, 196]

    def __len__(self):
        return len(self.voc_train)
    
    def class_id_to_name(self, i: int):
        return self.CLASS_IDX_TO_NAME[i]
    
    def setup(self, stage: Optional[str] = None):
        # train_bin_paths = [Path(self.data_dir) / str(bin) for bin in self.train_bins]
        val_bin_paths = [Path(self.data_dir) / str(class_num) / str(bin) 
                                                for class_num in self.classes 
                                                for bin in self.val_bins]
        
        if self.train_bins is not None:
            train_bin_paths = [Path(self.data_dir) / str(class_num) / str(bin) 
                                                for class_num in self.classes 
                                                for bin in self.train_bins]

            self.train_dataset = MVImgNetDataset(
                bin_paths=train_bin_paths,
                transforms=self.train_transforms,
                return_masks=self.return_masks,
            )
        else :
            self.train_dataset = []
            
        self.val_dataset = MVImgNetDataset(
            bin_paths=val_bin_paths,
            transforms=self.val_transforms,
            return_masks=self.return_masks,
        )

        print(f"✅ MVImgNet Loaded → Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          drop_last=self.drop_last, pin_memory=True)

    def get_train_dataset_size(self):
        return len(self.train_dataset)

    def get_val_dataset_size(self):
        return len(self.val_dataset)
    
    def get_num_classes(self):
        return len(self.CLASS_IDX_TO_NAME)
    

class MVImgNetDataset(Dataset):
    def __init__(
        self,
        bin_paths: List[Path],  # List of angle bin folders like ["folder/path/1", "folder/path/2", ...]
        transforms: Optional[Callable] = None,
        return_masks: bool = True,
    ):
        self.bin_paths = [Path(p) for p in bin_paths]
        self.transforms = transforms
        self.return_masks = return_masks

        self.images, self.masks = self._collect()

        # print(' images', self.images)
        # print('mask ', self.masks)


    def _collect(self) -> Tuple[List[Path], List[Optional[Path]]]:
        image_paths = []
        mask_paths = []

        for bin_path in self.bin_paths:
            img_dir = bin_path / "img"
            mask_dir = bin_path / "mask"

            if not img_dir.exists():
                print(f"⚠️ Missing image dir: {img_dir}")
                continue

            for img_file in sorted(img_dir.glob("*.jpg")):
                mask_file = mask_dir / f"{img_file.name}.png"
                if self.return_masks and not mask_file.is_file():
                    print(f"⚠️ Skipping due to missing mask: {mask_file}")
                    continue

                image_paths.append(img_file)
                mask_paths.append(mask_file if self.return_masks else None)
        assert all([Path(f).is_file() for f in mask_paths]) and all([Path(f).is_file() for f in image_paths])
        return image_paths, mask_paths

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]) if self.return_masks else None

        if self.transforms:
            if self.return_masks:
                img, mask = self.transforms(img, mask)
                
                mask = (mask > 0).float()  
                # ToDo: Should we use a different treshold? Note the borders of the objects are dropped for VOC 
                # and if we use a different treshold, we will keep some of them. We may then need to give a good reason for the treshold we chose.
                # mask[mask > 0.5] = 1
                # mask[mask <= 0.5] = 0

                path_to_img = self.images[index]
                if path_to_img.parent.parent.parent.name == '7':
                    mask = mask*1/ 255.0
                elif path_to_img.parent.parent.parent.name == '8':
                    mask = mask*2 / 255.0
                elif path_to_img.parent.parent.parent.name == '19':
                    mask = mask*3 / 255.0
                elif path_to_img.parent.parent.parent.name == '46':
                    mask = mask*4 / 255.0
                elif path_to_img.parent.parent.parent.name == '57':
                    mask = mask*5 / 255.0
                elif path_to_img.parent.parent.parent.name == '60':
                    mask = mask*6 / 255.0
                elif path_to_img.parent.parent.parent.name == '70':
                    mask = mask*7 / 255.0
                elif path_to_img.parent.parent.parent.name == '99':
                    mask = mask*8 / 255.0
                elif path_to_img.parent.parent.parent.name == '100':
                    mask = mask*9 / 255.0
                elif path_to_img.parent.parent.parent.name == '113':
                    mask = mask*10 / 255.0
                elif path_to_img.parent.parent.parent.name == '125':
                    mask = mask*11 / 255.0
                elif path_to_img.parent.parent.parent.name == '126':
                    mask = mask*12 / 255.0
                elif path_to_img.parent.parent.parent.name == '152':
                    mask = mask*13 / 255.0
                elif path_to_img.parent.parent.parent.name == '166':
                    mask = mask*14 / 255.0
                elif path_to_img.parent.parent.parent.name == '196':
                    mask = mask*15 / 255.0

                # mask = mask / 255.0  # This is done to avoid the object being ignored. See the create_memory() function for more details.
                # mask[mask == 1] = 0.99  # Or this?

            else:
                img = self.transforms(img)

        return (img, mask) if mask is not None else img

    def __len__(self) -> int:
        return len(self.images)

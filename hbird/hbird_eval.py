if __name__ == "__main__":
    # Add project root to path if running this file as a main
    import sys
    import pathlib
    p = str(pathlib.Path(__file__).parent.resolve()) + '/'
    sys.path.append(p)


import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(iterator, *args, **kwargs):
        return iterator

# import scann
import os
import random
import numpy as np
from hbird.models import FeatureExtractor
from hbird.models import FeatureExtractorSimple
from hbird.utils.eval_metrics import PredsmIoU
from hbird.utils.io import read_file_set

from hbird.utils.transforms import get_hbird_val_transforms, get_hbird_train_transforms, get_hbird_train_transforms_for_imgs

from hbird.utils.image_transformations import CombTransforms
from hbird.data.voc_data import VOCDataModule
from hbird.data.ade20k_data import Ade20kDataModule
from hbird.data.cityscapes_data import CityscapesDataModule
from hbird.data.coco_data import CocoDataModule
from hbird.data.mvimgnet_data import MVImgNetDataModule


class HbirdEvaluation():
    def __init__(self, feature_extractor, train_loader, n_neighbours, augmentation_epoch, num_classes, device, nn_method='scann', nn_params=None, memory_size=None, dataset_size=None, f_mem_p=None, l_mem_p=None):
        """
        Initializes the HbirdEvaluation class with the given parameters.

        Args:
            feature_extractor (torch.nn.Module): The feature extractor model.
            train_loader (DataLoader): DataLoader for the training dataset.
            n_neighbours (int): Number of nearest neighbors for NN search.
            augmentation_epoch (int): Number of augmentation epochs.
            num_classes (int): Number of classes in the dataset.
            device (str): Device to use ('cpu' or 'cuda').
            nn_method (str): Nearest neighbor search method ('scann' or 'faiss').
            nn_params (dict, optional): Additional parameters for NN search.
            memory_size (int, optional): Size of the memory for storing features.
            dataset_size (int, optional): Size of the dataset.
            f_mem_p (str, optional): Path to save feature memory.
            l_mem_p (str, optional): Path to save label memory.
        """
        if nn_params is None:
            nn_params = {}
        self.nn_params = nn_params
        self.feature_extractor = feature_extractor
        self.device = device
        self.nn_method = nn_method
        assert self.nn_method in ['faiss', 'scann'], "Only faiss and scann are supported"
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.n_neighbours = n_neighbours
        self.feature_extractor.eval()
        self.feature_extractor = feature_extractor.to(self.device)
        self.num_classes = num_classes
        eval_spatial_resolution = self.feature_extractor.eval_spatial_resolution
        self.num_sampled_features = None
        self.f_mem_p = f_mem_p
        self.l_mem_p = l_mem_p

        if self.memory_size is not None:
            self.num_sampled_features = self.memory_size // (dataset_size * self.augmentation_epoch)
            ## create memory of specific size
            self.feature_memory = torch.zeros((self.memory_size, self.feature_extractor.d_model))
            self.label_memory = torch.zeros((self.memory_size, self.num_classes ))
        self.create_memory(train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution)
        self.save_memory()
        self.feature_memory = self.feature_memory.cpu()
        self.label_memory = self.label_memory.cpu()
        self.create_NN(self.n_neighbours, nn_method=self.nn_method, **self.nn_params)

    def create_NN(self, n_neighbours=30, nn_method='faiss', **kwargs):
        """
        Creates a nearest neighbor search algorithm based on the specified method.

        Args:
            n_neighbours (int): Number of nearest neighbors to search for.
            nn_method (str): Nearest neighbor search method ('faiss' or 'scann').
            **kwargs: Additional parameters for the NN search algorithm.
        """
        if nn_method == 'scann':
            from hbird.nn.search_scann import NearestNeighborSearchScaNN
            self.NN_algorithm = NearestNeighborSearchScaNN(self.feature_memory, n_neighbors=n_neighbours, **kwargs)
        elif nn_method == 'faiss':
            from hbird.nn.search_faiss import NearestNeighborSearchFaiss
            self.NN_algorithm = NearestNeighborSearchFaiss(self.feature_memory, n_neighbors=n_neighbours, **kwargs)

    def create_memory(self, train_loader, num_classes, eval_spatial_resolution):
        """
        Creates a memory of features and labels from the training dataset.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            num_classes (int): Number of classes in the dataset.
            eval_spatial_resolution (int): Spatial resolution for evaluation.
        """
        feature_memory = list()
        label_memory = list()
        idx = 0
        with torch.no_grad():
            for j in tqdm(range(self.augmentation_epoch), desc='Augmentation loop', mininterval=10):
                for i, (x, y) in enumerate(tqdm(train_loader, desc='Memory Creation loop', mininterval=10)):
                    # Here x is the image; y is a the mask (full with values between [0, 1]
                    
                    x = x.to(self.device)
                    y = y.to(self.device)
                    y = (y * 255).long()  # y is scaled to be a values in [0, 255]
                    y[y == 255] = 0
                    # print(' unique y', torch.unique(y))

                    # Note that:
                    # In the masks of the VOC dataset, the borders are 255 (white), 
                    # the background is 0 (black), and the objects are grey.
                    # The following line ensures borders are treated as background.
                    # If using a dataset with different semantics of the color (e.g. MVImgNet), 
                    # ensure your object is not 255.
                    # [class of pixel in y] -> [one_hot_vector of the pixel]
                    # [ 0 ] -> [1, 0, ...]
                    # [ 1 ] -> [0, 1, ...]
                    # [255] -> [0,..., 1] 
                    # len(list)= -> len(list)=255

                    features, _ = self.feature_extractor.forward_features(x) # features of shape (BS, PS, D)
                    input_size = x.shape[-1]
                    patch_size = input_size // eval_spatial_resolution
                    patchified_gts = self.patchify_gt(y, patch_size) ## (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    label = one_hot_patch_gt.mean(dim=3)

                    # ToDo: Can we remove that?
                    # ---------------- debugging
                    # if i == 0 and j == 0:

                    #     print("CREATE MEM ")
                    #     # print('one_hot_patch_gt : ' , one_hot_patch_gt)
                    #     print('patchified_gts shape : ' , patchified_gts.shape)
                    #     print()
                    #     print('one_hot_patch_gt shape : ' , one_hot_patch_gt.shape)
                    # print('one_hot_patch_gt unique : ' ,  torch.unique(one_hot_patch_gt))
                    #     print('one_hot_patch_gt sum 0 : ' ,  torch.sum(one_hot_patch_gt[:, :, : , :, 0]))
                    #     print('one_hot_patch_gt sum 1 : ' ,  torch.sum(one_hot_patch_gt[:, :, : , :, 1]))
                    #     print()
                    #     # print('label : ', label)
                    #     print('label shape : ', label.shape)
                    #     print('label unique : ' ,  torch.unique(label))
                    #     print('label sum 0 : ' ,  torch.sum(label[:, :, : ,0]))
                    #     print('label sum 1 : ' ,  torch.sum(label[:, :, : ,1]))
                    # ----------------
                    
                    if self.memory_size is None:
                        # Memory Size is unbounded so we store all the features
                        normalized_features = features / torch.norm(features, dim=2, keepdim=True)
                        
                        normalized_features = normalized_features.flatten(0, 1)
                        label = label.flatten(0, 2)
                        feature_memory.append(normalized_features.detach().cpu())
                        label_memory.append(label.detach().cpu())
                    else:
                        # Memory Size is bounded so we need to select/sample some features only
                        sampled_features, sampled_indices = self.sample_features(features, patchified_gts)
                        normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=2, keepdim=True)
                        label = label.flatten(1, 2)
                        ## select the labels of the sampled features
                        sampled_indices = sampled_indices.to(self.device)
                        ## repeat the label for each sampled feature
                        label_hat = label.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]))

                        # label_hat = label.gather(1, sampled_indices)
                        normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                        label_hat = label_hat.flatten(0, 1)
                        self.feature_memory[idx:idx+normalized_sampled_features.size(0)] = normalized_sampled_features.detach().cpu()
                        self.label_memory[idx:idx+label_hat.size(0)] = label_hat.detach().cpu()
                        idx += normalized_sampled_features.size(0)
                        # memory.append(normalized_sampled_features.detach().cpu())
            if self.memory_size is None:
                self.feature_memory = torch.cat(feature_memory)
                self.label_memory = torch.cat(label_memory)

    def save_memory(self):
        """
        Saves the feature and label memory to the specified file paths.
        """
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)

    def load_memory(self):
        """
        Loads the feature and label memory from the specified file paths.

        Returns:
            bool: True if memory is successfully loaded, False otherwise.
        """
        if self.f_mem_p is not None and self.l_mem_p is not None and os.path.isfile(self.f_mem_p) and os.path.isfile(self.l_mem_p):
            self.feature_memory = torch.load(self.f_mem_p)
            self.label_memory = torch.load(self.l_mem_p)
            return True
        return False

    def sample_features(self, features, pathified_gts):
        """
        Samples features and their corresponding indices based on patch scores.

        Args:
            features (torch.Tensor): Feature tensor of shape (BS, PS, D).
            pathified_gts (torch.Tensor): Patchified ground truth tensor.

        Returns:
            tuple: Sampled features and their indices.
        """
        sampled_features = []
        sampled_indices = []
        for k, gt in enumerate(tqdm(pathified_gts, mininterval=10)):
            class_frequency = self.get_class_frequency(gt)
            patch_scores, nonzero_indices = self.get_patch_scores(gt, class_frequency)

            patch_scores = patch_scores.flatten()
            nonzero_indices = nonzero_indices.flatten()

            # assert zero_score_idx[0].size(0) != 0 ## for pascal every patch should belong to one class
            patch_scores[~nonzero_indices] = 1e6

            # sample uniform distribution with the same size as the
            # number of nonzero indices (we use the sum here as the
            # nonzero_indices matrix is a boolean mask)
            uniform_x = torch.rand(nonzero_indices.sum())
            patch_scores[nonzero_indices] *= uniform_x
            feature = features[k]

            ### select the least num_sampled_features score indices
            _, indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)

            sampled_indices.append(indices)
            samples = feature[indices]
            sampled_features.append(samples)

        sampled_features = torch.stack(sampled_features)
        sampled_indices = torch.stack(sampled_indices)

        return sampled_features, sampled_indices

    def get_class_frequency(self, gt):
        """
        Computes the frequency of each class in the ground truth.

        Args:
            gt (torch.Tensor): Ground truth tensor.

        Returns:
            torch.Tensor: Class frequency tensor.
        """
        class_frequency = torch.zeros((self.num_classes), device=self.device)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()
                class_frequency[patch_classes] += 1

        return class_frequency

    def get_patch_scores(self, gt, class_frequency):
        """
        Computes patch scores based on class frequency.

        Args:
            gt (torch.Tensor): Ground truth tensor.
            class_frequency (torch.Tensor): Class frequency tensor.

        Returns:
            tuple: Patch scores and nonzero indices.
        """
        patch_scores = torch.zeros((gt.shape[0], gt.shape[1]))
        nonzero_indices = torch.zeros((gt.shape[0], gt.shape[1]), dtype=torch.bool)

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                patch_classes = gt[i, j].unique()
                patch_scores[i, j] = class_frequency[patch_classes].sum()
                nonzero_indices[i, j] = patch_classes.shape[0] > 0

        return patch_scores, nonzero_indices

    def patchify_gt(self, gt, patch_size):
        """
        Converts the ground truth into patches.

        Args:
            gt (torch.Tensor): Ground truth tensor.
            patch_size (int): Size of each patch.

        Returns:
            torch.Tensor: Patchified ground truth tensor.
        """
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h//patch_size, patch_size, w//patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h//patch_size, w//patch_size, c*patch_size*patch_size)
        return gt

    def cross_attention(self, q, k, v, beta=0.02):
        """
        Computes cross-attention between query, key, and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (BS, num_patches, d_k).
            k (torch.Tensor): Key tensor of shape (BS, num_patches, NN, d_k).
            v (torch.Tensor): Value tensor of shape (BS, num_patches, NN, label_dim).
            beta (float): Scaling factor for attention computation.

        Returns:
            torch.Tensor: Attention-weighted labels.
        """
        d_k = q.size(-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2) ## (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat
    
    def find_nearest_key_to_query(self, q):
        """
        Finds the nearest key features and labels for a given query.

        Args:
            q (torch.Tensor): Query tensor of shape (BS, num_patches, d_k).

        Returns:
            tuple: Nearest key features and labels.
        """
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs*num_patches, d_k)
        # neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors, distances = self.NN_algorithm.find_nearest_neighbors(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).cpu()
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.n_neighbours, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.n_neighbours, -1)
        return key_features, key_labels

    def evaluate(self, val_loader, eval_spatial_resolution, return_knn_details=False, ignore_index=255):
        """
        Evaluates the model on the validation dataset.

        Args:
            val_loader (DataLoader): DataLoader for the validation dataset.
            eval_spatial_resolution (int): Spatial resolution for evaluation.
            return_knn_details (bool): Whether to return KNN details.
            ignore_index (int): Index to ignore during evaluation.

        Returns:
            float or tuple: Evaluation metric (e.g., Jaccard index) and optionally KNN details.
        """
        metric = PredsmIoU(self.num_classes, self.num_classes)
        self.feature_extractor = self.feature_extractor.to(self.device)
        label_hats = []
        lables = []

        knns = []
        knns_labels = []
        knns_ca_labels = []
        idx = 0

        input_lis = []  # list to store input images
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader, desc='Evaluation loop')):
                x = x.to(self.device)

                input_lis.append(x)
                _, _, h, w = x.shape
                features, _ = self.feature_extractor.forward_features(x)
                features = features.cpu()
                y = (y * 255).long()
                ## copy the data of features to another variable
                q = features.clone().detach()
                key_features, key_labels = self.find_nearest_key_to_query(q)           
                label_hat = self.cross_attention(features, key_features, key_labels)

                # ToDo: Can remove that?
                # ----------------- debugging
                # # x is the input image
                # # y is the input mask 
                # if i == 0:
                #     print("PART 0")

                #     print('feauture:', self.feature_memory)
                #     print('feauture unique:', torch.unique(self.feature_memory))
                #     print('feautureshape:', self.feature_memory.shape)

                #     print('memory label :', self.label_memory)
                #     print('memory label unique:', torch.unique(self.label_memory))
                #     print('memory label shape:', self.label_memory.shape)
                #     print('memory label sum 0:', torch.sum(self.label_memory[:, 0]))
                #     print('memory label sum 1:', torch.sum(self.label_memory[:, 1]))
                #     print()
                #     print()
                #     print('input image :' , x)
                #     print('input image shape :' , x.shape)
                #     print('input image unique :' , torch.unique(x))

                #     print('mask  :' , y)
                #     print('mask shape :' , y.shape)
                #     print('mask unique :' , torch.unique(y))

                #     print()
                #     print('key_labels unique: ', np.unique(key_labels))
                #     print('key_labels shape : ', key_labels.shape)

                #     print('key_features unique  : ', np.unique(key_features))
                #     print('key_features shape : ', key_features.shape)

                #     print('q  unique : ', np.unique(q))
                #     print('q shape : ', q.shape)

                #     print('label hat unique : ', np.unique(label_hat))
                #     print('label hat shape : ', label_hat.shape)
                #     print()
                # ---------------------

                if return_knn_details:
                    knns.append(key_features.detach())
                    knns_labels.append(key_labels.detach())
                    knns_ca_labels.append(label_hat.detach())
                bs, _, label_dim = label_hat.shape
                label_hat = label_hat.reshape(bs, eval_spatial_resolution, eval_spatial_resolution, label_dim).permute(0, 3, 1, 2)
                resized_label_hats =  F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
                # ToDo: overlay clustermap them with GT
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1)
                label_hats.append(cluster_map.detach())
                lables.append(y.detach())
                    
            # The full labels and label hats (no class ignored)
            # ToDo: Check if we can remove that. No class is ignored, ignore_index=-1 is set
            # full_label_hats = label_hats
            # full_lables = lables

            lables = torch.cat(lables) 
            label_hats = torch.cat(label_hats)
            valid_idx = lables != ignore_index
            valid_target = lables[valid_idx]
            valid_cluster_maps = label_hats[valid_idx]

            # ToDo: Check if we can remove that (no object will be 255, see mvimgnet_dataset)
            # # valid_cluster_maps[valid_cluster_maps == 255] = 1
            # valid_cluster_maps.masked_fill_(valid_cluster_maps == 255, 1)
            # # valid_target[valid_target == 255] = 1
            # valid_target.masked_fill_(valid_target == 255, 1)

            metric.update(valid_target, valid_cluster_maps)
            jac, tp, fp, fn, reordered_preds, matched_bg_clusters = metric.compute(is_global_zero=True)

            # ToDo: Check if we can remove that
            # return full_label_hats, full_lables, jac, input_lis

            if return_knn_details:
                knns = torch.cat(knns)
                knns_labels = torch.cat(knns_labels)
                knns_ca_labels = torch.cat(knns_ca_labels)
                return jac, {"knns": knns, "knns_labels": knns_labels, "knns_ca_labels": knns_ca_labels}
            else:
                return jac

def hbird_evaluation(model, d_model, patch_size, dataset_name:str, data_dir:str, batch_size=64, input_size=224, 
                        augmentation_epoch=1, device='cpu', return_knn_details=False, n_neighbours=30, nn_method='scann', nn_params=None, 
                        ftr_extr_fn=None, memory_size=None, num_workers=8, ignore_index=255, train_fs_path=None, val_fs_path=None, train_bins=None, val_bins=None):
    """
    Performs evaluation of the Hbird model on a specified dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        d_model (int): Dimensionality of the model's feature space.
        patch_size (int): Size of patches for processing.
        dataset_name (str): Name of the dataset to evaluate on.
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for data loading.
        input_size (int): Input size for the model.
        augmentation_epoch (int): Number of augmentation epochs.
        device (str): Device to use ('cpu' or 'cuda').
        return_knn_details (bool): Whether to return KNN details.
        n_neighbours (int): Number of nearest neighbors for NN search.
        nn_method (str): Nearest neighbor search method ('scann' or 'faiss').
        nn_params (dict, optional): Additional parameters for NN search.
        ftr_extr_fn (callable, optional): Custom feature extraction function.
        memory_size (int, optional): Size of the memory for storing features.
        num_workers (int): Number of workers for data loading.
        ignore_index (int): Index to ignore during evaluation.
        train_fs_path (str, optional): Path to the training file set.
        val_fs_path (str, optional): Path to the validation file set.
        train_bins (str, optional): Training angle bins for MVImgNet.
        val_bins (str, optional): Validation angle bins for MVImgNet.

    Returns:
        float or tuple: Evaluation metric (e.g., Jaccard index) and optionally KNN details.
    """
    eval_spatial_resolution = input_size // patch_size

    if ftr_extr_fn is None:
        feature_extractor = FeatureExtractor(model, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    else:
        feature_extractor = FeatureExtractorSimple(model, ftr_extr_fn=ftr_extr_fn, eval_spatial_resolution=eval_spatial_resolution, d_model=d_model)
    train_transforms_dict = get_hbird_train_transforms(input_size)
    val_transforms_dict = get_hbird_val_transforms(input_size)

    train_transforms = CombTransforms(img_transform=train_transforms_dict['img'], tgt_transform=None, img_tgt_transform=train_transforms_dict['shared'])
    val_transforms = CombTransforms(img_transform=val_transforms_dict['img'], tgt_transform=None, img_tgt_transform=val_transforms_dict['shared'])
    
    dataset_size = 0
    num_classes = 0
    ignore_index = -1 

    train_file_set=None

    if train_fs_path is not None:
        train_file_set = read_file_set(train_fs_path)
    val_file_set=None
    if val_fs_path is not None:
        val_file_set = read_file_set(val_fs_path)
    
    # Setup the dataset
    sample_fract=None
    if "*" in dataset_name:
        parts = dataset_name.split("*")
        dataset_name = parts[0]
        sample_fract = float(parts[1])
        print(f"Using {sample_fract} fraction of the {dataset_name} dataset.")

    if dataset_name == "voc":
        # Pascal VOC dataset requires always a file set for training and validation
        if train_file_set is None:
            train_file_set = read_file_set(os.path.join(data_dir, "sets", "trainaug.txt"))
        if val_file_set is None:
            val_file_set = read_file_set(os.path.join(data_dir, "sets", "val.txt"))

        if sample_fract is not None:
            random.shuffle(train_file_set)
            train_file_set = train_file_set[:int(len(train_file_set)*sample_fract)]
            print(f"sampled {len(train_file_set)} Pascal VOC images for training")

        ignore_index = 255
        dataset = VOCDataModule(batch_size=batch_size,
                                    num_workers=num_workers,
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_dir,
                                    train_image_transform=train_transforms,
                                    val_transforms=val_transforms,
                                    shuffle=False,
                                    return_masks=True,
                                    train_file_set=train_file_set,
                                    val_file_set=val_file_set)
        dataset.setup()
    elif dataset_name == "ade20k":

        if sample_fract is not None:
            if train_file_set is None:
                # if the train_file_set is not provided, we sample from the whole dataset
                train_file_set = [f.replace(".jpg","") for f in os.listdir(os.path.join(data_dir, 'images','training'))]
            random.shuffle(train_file_set)
            train_file_set = train_file_set[:int(len(train_file_set)*sample_fract)]
            print(f"sampled {len(train_file_set)} ADE20k images for training")

        ignore_index = 0
        dataset = Ade20kDataModule(data_dir,
                 train_transforms=train_transforms,
                 val_transforms=val_transforms,
                 shuffle=False,
                 num_workers=num_workers,
                 batch_size=batch_size,
                 train_file_set=train_file_set,
                 val_file_set=val_file_set)
        dataset.setup()
    elif dataset_name == "cityscapes":

        if sample_fract is not None:
            if train_file_set is None:
                img_folder = os.path.join(data_dir, 'leftImg8bit', 'train')
                train_file_set=list()
                for root, _, files in os.walk(img_folder):
                    for filename in files:
                        if filename.endswith('.png'):
                            base_name = filename.split("_leftImg8bit.png")[0]
                            train_file_set.append(base_name)
            random.shuffle(train_file_set)
            train_file_set = train_file_set[:int(len(train_file_set)*sample_fract)]
            print(f"sampled {len(train_file_set)} Cityscapes images for training")

        ignore_index = 255
        dataset = CityscapesDataModule(root=data_dir,
                                           train_transforms=train_transforms,
                                           val_transforms=val_transforms,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           batch_size=batch_size,
                                           train_file_set=train_file_set,
                                           val_file_set=val_file_set)
        dataset.setup()
    elif "coco" in dataset_name:
        assert len(dataset_name.split("-")) == 2
        mask_type = dataset_name.split("-")[-1]
        assert mask_type in ["thing", "stuff"]
        if mask_type == "thing":
            num_classes = 12
        else:
            num_classes = 15
        ignore_index = 255

        if sample_fract is not None:
            if train_file_set is None:
                # if the train_file_set is not provided, we sample from the whole dataset
                train_file_set = os.listdir(os.path.join(data_dir, "images", "train2017"))
            random.shuffle(train_file_set)
            train_file_set = train_file_set[:int(len(train_file_set)*sample_fract)]
            print(f"sampled {len(train_file_set)} COCO images for training")

        dataset = CocoDataModule(batch_size=batch_size,
                                     num_workers=num_workers,
                                     data_dir=data_dir,
                                     mask_type=mask_type,
                                     train_transforms=train_transforms,
                                     val_transforms=val_transforms,
                                     train_file_set=train_file_set,
                                     val_file_set=val_file_set)
        dataset.setup()
    elif dataset_name == "mvimgnet":
        dataset = MVImgNetDataModule(
            data_dir=data_dir,
            train_bins=train_bins,
            val_bins=val_bins,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            return_masks=True,
        )
        dataset.setup()
        # The default ignore_index=-1 is used to not ignore any class
    else:
        raise ValueError("Unknown dataset name")

    
    # Evaluate the model
    if dataset_name == "mvimgnet":  # evaluation is done on a specific bin for all classes
        # Build the memeory
        dataset_size = dataset.get_train_dataset_size()
        num_classes = dataset.get_num_classes()  # the num classes is the same for training and validation
        train_loader = dataset.train_dataloader()
        # ToDo: check if we can remove that
        # evaluator = HbirdEvaluation(feature_extractor, train_loader, n_neighbours=n_neighbours, 
        #                     augmentation_epoch=augmentation_epoch, num_classes=num_classes, 
        #                     device=device, nn_method=nn_method, nn_params=nn_params, memory_size=memory_size, 
        #                     dataset_size=dataset_size)

        # Evaluate on each of the val_bins separately (val_loader and val_bin_dataset are needed for each bin)
        miou_list = []
        for val_bin in val_bins:
            val_bin_dataset = MVImgNetDataModule(
                data_dir=data_dir,
                train_bins=None,
                val_bins=[val_bin],
                train_transforms=train_transforms,
                val_transforms=val_transforms,
                batch_size=batch_size,
                num_workers=num_workers,
                return_masks=True,
            )
            val_bin_dataset.setup()
            # val_bin_dataset_size = dataset.get_train_dataset_size()  # ToDo: Do we need that?
            val_bin_loader = val_bin_dataset.val_dataloader()
            evaluator = HbirdEvaluation(feature_extractor, train_loader, n_neighbours=n_neighbours, 
                                augmentation_epoch=augmentation_epoch, num_classes=num_classes, 
                                device=device, nn_method=nn_method, nn_params=nn_params, memory_size=memory_size, 
                                dataset_size=dataset_size)  # ToDo: check if we should pass dataset_size/val_bin_dataset_size
            
            val_bin_miou = evaluator.evaluate(
                val_bin_loader,
                eval_spatial_resolution,
                return_knn_details=return_knn_details,
                ignore_index=ignore_index  # the default ignore_index=-1 is used
                )
            miou_list.append(val_bin_miou)
            print(f" train_bins: {train_bins}, val_bin : {val_bin}, mIoU for this val_bin: {val_bin_miou}, mean mIoU : {np.mean(miou_list)}")

        return miou_list  # list of mIoU for the val_bin-s

    else:  # for all other datasets evaluation is done once on a single validation set
        # Build the memeory and evaluate on the validation set
        dataset_size = dataset.get_train_dataset_size()
        num_classes = dataset.get_num_classes()
        train_loader = dataset.train_dataloader()
        val_loader = dataset.val_dataloader()
        evaluator = HbirdEvaluation(feature_extractor, train_loader, n_neighbours=n_neighbours, 
                            augmentation_epoch=augmentation_epoch, num_classes=num_classes, 
                            device=device, nn_method=nn_method, nn_params=nn_params, memory_size=memory_size, 
                            dataset_size=dataset_size)
        return evaluator.evaluate(
            val_loader, 
            eval_spatial_resolution, 
            return_knn_details=return_knn_details, 
            ignore_index=ignore_index
            )
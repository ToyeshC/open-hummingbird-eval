import os
import json
import torch
import random
import argparse
import numpy as np
import csv

from hbird.hbird_eval import hbird_evaluation


def seed_everything(seed: int):
    """Ensure reproducibility across torch, numpy, and python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_model(args):
    """
    Load model based on the source:
    - DINO from torch.hub
    - MoCo from HuggingFace (ResNet-50 backbone)
    - Transformers from HuggingFace
    """
    repo = args.model_repo

    if '/' in repo and not repo.startswith('facebookresearch/'):
        from transformers import AutoModel
        print(f"Loading transformer model from HF Hub: {repo}")
        model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        model.eval()
        return model
    
    if 'moco' in repo.lower():
        from torchvision.models import resnet50
        from huggingface_hub import hf_hub_download

        model = resnet50(pretrained=False)

        ckpt_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_filename,
            cache_dir=os.path.expanduser('~/.cache/moco'),
            local_dir_use_symlinks=False,
        )
        print(f"Loaded MoCo checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        cleaned = {k.replace('module.encoder_q.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=True)
        model.eval()
        return model
    
    if 'clip' in args.model_repo.lower():
        from transformers import CLIPModel
        print(f"Loading CLIP model from HF Hub: {repo}")
        model = CLIPModel.from_pretrained(repo)
        model.eval()
        return model

    # Default: DINO via torch.hub
    print(f"Loading model via torch.hub: {repo}, {args.model_name}")
    model = torch.hub.load(repo, args.model_name, pretrained=True)
    model.eval()
    return model


def main(args):
    print(f"The script arguments are {args}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args).to(device)

    # Parse --nn_params if provided, else use empty dict
    if args.nn_params:
        try:
            nn_params = json.loads(args.nn_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid format for --nn_params. Provide a valid JSON string.")
    else:
        nn_params = {}

    # # Decide whether to enable FAISS sharding (this might help with out of memory errors)
    # num_gpus = torch.cuda.device_count()
    # should_use_sharding = (
    #     device == "cuda"
    #     and args.nn_method == "faiss"
    #     and num_gpus > 1
    # )
    # if should_use_sharding:
    #     print(f"Detected {num_gpus} GPUs. Enabling FAISS index sharding.")
    #     nn_params.setdefault("idx_shard", True)
    # else:
    #     print(f"FAISS sharding not used (device: {device}, GPUs available: {num_gpus})")


    # Handle MVImgNet angle bin parsing
    if args.dataset_name.lower() == "mvimgnet":
        assert args.train_bins is not None, "You must specify --train_bins for mvimgnet."
        assert args.val_bins is not None, "You must specify --val_bins for mvimgnet."

        train_bins_list = args.train_bins.split(',')
        val_bins_list = args.val_bins.split(',')

        assert type(train_bins_list) == list, "train_bins must be a comma-separated list."
        assert type(val_bins_list) == list, "val_bins must be a comma-separated list."

        print(f"📦 MVImgNet → Train bins: {train_bins_list}")
        print(f"📦 MVImgNet → Val bins:   {val_bins_list}")


    # Define feature extractor hook
    def token_features(model, imgs):
        if "moco" in args.model_repo.lower():
            return model(imgs), None
        elif "dinov2" in args.model_repo.lower():
            return model.forward_features(imgs)['x_norm_patchtokens'], None
        elif "clip" in args.model_repo.lower():
            # Get last hidden states from vision model
            vision_outputs = model.vision_model(pixel_values=imgs, output_hidden_states=True)
            last_hidden = vision_outputs.hidden_states[-1]  # [B, num_tokens, D]
            return last_hidden[:, 1:], None  # Exclude CLS
        elif "siglip" in args.model_repo.lower():
            # Get last hidden states from vision model
            vision_outputs = model.vision_model(pixel_values=imgs, output_hidden_states=True)
            last_hidden = vision_outputs.hidden_states[-1]  # [B, num_tokens, D]
            return last_hidden[:, 1:], None  # Exclude CLS
        elif "radio" in args.model_repo.lower():
            summary, spatial_features = model(imgs)
            return spatial_features, None
        elif "webssl" in args.model_repo.lower():
            outputs = model(pixel_values=imgs, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state
            return last_hidden[:, 1:], None  # Exclude CLS
        else:
            return model.get_intermediate_layers(imgs)[0][:, 1:], None  # CLS token excluded

    # Run Hummingbird evaluation
    # todo this is very inefficient for large memory banks (for mvimgnet its fine)
    # print(val_bins_list)
    # for val_bin in val_bins_list:

    # fn = lambda model, imgs: (model.get_intermediate_layers(imgs)[0][:, 1:], None)
    # hbird_miou = hbird_evaluation(
    #     model=model,
    #     d_model=args.d_model,
    #     patch_size=args.patch_size,
    #     batch_size=args.batch_size,
    #     input_size=args.input_size,
    #     augmentation_epoch=args.augmentation_epoch,
    #     device=device,
    #     return_knn_details=args.return_knn_details,
    #     nn_method=args.nn_method,
    #     n_neighbours=args.n_neighbours,
    #     nn_params=nn_params,
    #     ftr_extr_fn=token_features,
    #     dataset_name=args.dataset_name,
    #     data_dir=args.data_dir,
    #     memory_size=args.memory_size,
    #     num_workers=args.num_workers,
    #     ignore_index=-1, # added
    #     train_fs_path=args.train_fs_path,
    #     val_fs_path=args.val_fs_path,
    #     # added
    #     train_bins=train_bins_list,
    #     val_bins=val_bins_list,
    # )
    # print(f"val_bin(s) : {val_bins_list},  Hummingbird Evaluation (mIoU): {hbird_miou}")

    # FOR ALL PERMUTATIONS
    from itertools import permutations
    # import numpy as np

    # job_id = os.environ.get('SLURM_JOB_ID')

    # bins = [0, 15, 30, 45, 60, 75, 90]
    # lengths = range(1, 8)  

    # # Use set union in a loop
    # all_sets = set()

    # for r in lengths:
    #     perms = permutations(bins, r)
    #     sets_r = {frozenset(p) for p in perms}
    #     all_sets.update(sets_r)

    # results_path = "results/results_exp_b.csv"
    # if not os.path.exists(results_path):
    #     os.makedirs("results", exist_ok=True)
    #     with open(results_path, "w") as f:
    #         f.write("job_id,model,class,train_bins,val_bin,jac0,jac1,jac_mean\n")

    # for train_bin in list(all_sets):
    #     train_bin = list(train_bin)        
    #     hbird_miou = hbird_evaluation(
    #         model=model,
    #         d_model=args.d_model,
    #         patch_size=args.patch_size,
    #         batch_size=args.batch_size,
    #         input_size=args.input_size,
    #         augmentation_epoch=args.augmentation_epoch,
    #         device=device,
    #         return_knn_details=args.return_knn_details,
    #         nn_method=args.nn_method,
    #         n_neighbours=args.n_neighbours,
    #         nn_params=nn_params,
    #         ftr_extr_fn=token_features,
    #         dataset_name=args.dataset_name,
    #         data_dir=f"{args.data_dir}/{args.class_num}",
    #         memory_size=args.memory_size,
    #         num_workers=args.num_workers,
    #         ignore_index=-1, # added
    #         train_fs_path=args.train_fs_path,
    #         val_fs_path=args.val_fs_path,
    #         # added
    #         train_bins=train_bin,
    #         val_bins=val_bins_list,
    #     )

    #     train_str = '_'.join(str(x) for x in sorted(train_bin))
    #     for i in range(len(bins)):
    #         with open(results_path, mode='a', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerow([job_id, args.model_name, args.class_num, train_str, bins[i], hbird_miou[i][0], hbird_miou[i][1], np.mean(hbird_miou[i])])
    #             print(f"train_bin : {train_str} , 0 preds acc : {round(hbird_miou[i][0], 2)}, 1 preds acc : {round(hbird_miou[i][1],2)}") 
    
    # pred_mask, gt_mask, jac, input_lis = hbird_miou
    # print(np.mean(jac))
    # save_data = {
    #     'pred_mask' : pred_mask,
    #     'gt_mask' : gt_mask,
    #     'jac' : jac,
    #     'input' : input_lis
    # }

    # torch.save(save_data, 'tensor_list.pt')





    # PASCAL
    hbird_miou = hbird_evaluation(model.to(device),
        d_model=args.d_model,        # size of the embedding feature vectors of patches
        patch_size=args.patch_size,
        batch_size = args.batch_size,
        input_size=args.input_size,
        augmentation_epoch=1,     # how many iterations of augmentations to use on top of the training dataset in order to generate the memory
        device=device,
        return_knn_details=False, # whether to return additional NNs details
        nn_method='faiss',
        n_neighbours=30,         # the number of neighbors to fetch per image patch
        nn_params=None,           # Other parameters to be used for the k-NN operator
        ftr_extr_fn=token_features,           # function that extracts features from a vision encoder on images
        dataset_name=args.dataset_name,       # the name of the dataset to use, currently only Pascal VOC is included.
        data_dir=args.data_dir,    # path to the dataset to use for evaluation
        memory_size=args.memory_size,
        train_fs_path=args.train_fs_path,
        val_fs_path=args.val_fs_path,
        num_workers=2,  # Google Colab says it can only support 2 workers. This fixes a warning and a session crash. The default is 8.
        )
    print(f"val_bin(s) :,  Hummingbird Evaluation (mIoU): {hbird_miou}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Reproducibility
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

    # # Device
    # parser.add_argument("--device", default="cuda", type=str, help="Device to run the model on")  # This is automatically set to cuda, don't pass

    # Model arguments
    parser.add_argument("--model_repo", default=None, type=str,
                        help="Torch Hub repo or HuggingFace repo ID")
    parser.add_argument("--model_name", default=None, type=str,
                        help="Model name for torch.hub (e.g. dino_vits16)")
    parser.add_argument("--d_model", default=None, type=int,
                        help="Size of the embedding feature vectors")

    # MoCo-specific args
    parser.add_argument("--hf_repo", default=None, type=str,
                        help="HuggingFace repo ID for MoCo checkpoint")
    parser.add_argument("--hf_filename", default=None, type=str,
                        help="Checkpoint filename in HuggingFace repo")
    # parser.add_argument("--hf_repo", default="facebook/moco-v2-checkpoints", type=str,
    #                     help="HuggingFace repo ID for MoCo checkpoint")
    # parser.add_argument("--hf_filename", default="moco_v2_800ep_pretrain.pth.tar", type=str,
    #                     help="Checkpoint filename in HuggingFace repo")

    # Input & patching
    parser.add_argument("--input_size", default=None, type=int, help="Size of the input image")
    parser.add_argument("--patch_size", default=None, type=int, help="Size of the model patch")

    # Dataset arguments
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Path to the dataset root")
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset name (e.g. voc, mvimgnet)")
    parser.add_argument("--train_fs_path", default=None, type=str,
                        help="Path to train file list")
    parser.add_argument("--val_fs_path", default=None, type=str,
                        help="Path to validation file list")
    
    parser.add_argument("--train_bins", type=str, default=None,
                        help="(MVImgNet only) Training angle bins as comma-sep list like 0,15,30")
    parser.add_argument("--val_bins", type=str, default=None,
                        help="(MVImgNet only) Validation angle bins as comma-sep list like 0,15,30")
    parser.add_argument("--class_num", type=str, help="(MVImgNet only) object class")
    
    # Evaluation behavior
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for evaluation")
    parser.add_argument("--augmentation_epoch", default=1, type=int,
                        help="Number of augmentation passes over training data")
    parser.add_argument("--memory_size", default=None, type=int,
                        help="Optional memory size cap")
    parser.add_argument("--num_workers", default=None, type=int, help="Num workers for DataLoader")

    # Nearest neighbor search
    parser.add_argument("--n_neighbours", default=30, type=int,
                        help="Number of neighbors to use in k-NN search")
    parser.add_argument("--nn_method", default="faiss", type=str,
                        help="Method for nearest neighbor search")
    parser.add_argument("--nn_params", default=None, type=str,
                        help="JSON string for nearest neighbor parameters")
    parser.add_argument("--return_knn_details", action="store_true",
                        help="Whether to return details of k-NN results")

    parser.add_argument("--job_id", default=None,
                        help="job_id of job")

    args = parser.parse_args()

    seed_everything(args.seed)

    main(args)

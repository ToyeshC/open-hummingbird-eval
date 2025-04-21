import argparse
import torch
from hbird.hbird_eval import hbird_evaluation
import json


def get_args():
    parser = argparse.ArgumentParser(description="Hummingbird Evaluation")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument('--model_repo', default='facebookresearch/dino:main', type=str, help='Repo for torch.hub.load')
    parser.add_argument('--model_name', default='dino_vits16', type=str, help='Model name for torch.hub.load')
    parser.add_argument('--dataset_name', default='voc', type=str)
    parser.add_argument('--data_dir', default='./datasets/TinyVOCSegmentation', type=str)
    parser.add_argument('--train_fs_path', default='./datasets/TinyVOCSegmentation/sets/trainaug.txt', type=str)
    parser.add_argument('--val_fs_path', default='./datasets/TinyVOCSegmentation/sets/val.txt', type=str)
    parser.add_argument('--n_neighbours', default=30, type=int)
    parser.add_argument('--augmentation_epoch', default=1, type=int)
    parser.add_argument('--return_knn_details', action='store_true')
    parser.add_argument('--nn_method', default='faiss', type=str)
    parser.add_argument('--memory_size', default=None, type=int)
    parser.add_argument('--nn_params', default=None, type=str)  # assume stringified JSON or dict
    return parser.parse_args()


def main():
    args = get_args()

    model = torch.hub.load(args.model_repo, args.model_name, pretrained=True)
    model = model.to(args.device)

    # Convert nn_params string to dict if needed
    if args.nn_params:
        try:
            nn_params = json.loads(args.nn_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid format for --nn_params. Please pass a valid JSON string.")
    else:
        nn_params = None

    fn = lambda model, imgs: (model.get_intermediate_layers(imgs)[0][:, 1:], None)

    hbird_miou = hbird_evaluation(
        model,
        d_model=args.embed_dim,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        input_size=args.input_size,
        augmentation_epoch=args.augmentation_epoch,
        device=args.device,
        return_knn_details=args.return_knn_details,
        nn_method=args.nn_method,
        n_neighbours=args.n_neighbours,
        nn_params=nn_params,
        ftr_extr_fn=fn,
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        memory_size=args.memory_size,
        train_fs_path=args.train_fs_path,
        val_fs_path=args.val_fs_path,
    )

    print('Hummingbird Evaluation (mIoU):', hbird_miou)


if __name__ == "__main__":
    main()

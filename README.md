### Description
This repository is a reproduction repository that implements the Dense NN Retrieval Evaluation method introduced by Balažević et al. ["Towards In-context Scene Understanding", NeurIPS 2023](https://arxiv.org/abs/2306.01667).

Briefly, it evaluates the effectiveness of spatial features acquired from a vision encoder, to associate themselves to relevant features from a dataset (validation), through the utilization of a k-NN classifier/retriever that operates across various proportions of training data.


![Hummingbird Evaluation](./images/hbird_icl_diagram.png)
Image taken from ["Towards In-context Scene Understanding", NeurIPS 2023](https://arxiv.org/abs/2306.01667).

This evaluation approach helps understand scenes by comparing new images with ones we already know. We start by showing it a bunch of densely labeled images. It densely encodes the images such that we have both the encoded patches (top-left section) and their labels (bottom-left section) as taken from a set of image-label examples given (left part). Then, we give it new images to describe (right part) without the labels, which again densely encodes. Then, it compares parts (encoded patches) of each of the given images with similar parts in the examples it knows. By looking at what's closest, it figures out what is the potential label for that part and therefore on what the new image might be showing. This is a flexible approach because it doesn't assume anything about the labels.

Reproduction done by:
* Valentinos Pariza
* Mohammadreza Salehi
* Yuki M. Asano

At the **University of Amsterdam (UvA)**


### Notes
* For any questions/issues etc. please open a github issue on this repository.
* If you find this repository useful, please consider starring and citing.

### Results we got with our implementation on Pascal VOC
For the experiments below we used two dataset augmentation epochs and
also we used image size of (512,512) for the dino and (504,504) for dinov2.

<table>
  <tr>
    <th rowspan="2">arch</th>
    <th rowspan="2">model</th>
    <th colspan="3">PVOC (mIoU) per Memory Size</th>
    <th colspan="1">PVOC (mIoU) <br> from orig. Paper</th>
  </tr>
  <tr>
    <th>1024*10<sup>2</sup></th>
    <th>1024*10<sup>3</sup></th>
    <th>1024*10<sup>4</sup></th>
    <th>1024*10<sup>4</sup></th>
  <tr>
    <td>ViT-S/16</td>
    <td>dino</td>
    <td>37.2</td>
    <td>43.1</td>
    <td>46.6</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>dino</td>
    <td>44.9</td>
    <td>50.8</td>
    <td>55.7</td>
    <td>55.9</td>
  </tr>
  <tr>
    <td>ViT-S/14</td>
    <td>dinov2</td>
    <td>70.2</td>
    <td>74.9</td>
    <td>77.0</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ViT-B/14</td>
    <td>dinov2</td>
    <td>69.1</td>
    <td>74.6</td>
    <td>76.9</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ViT-L/14</td>
    <td>dinov2</td>
    <td>64.6</td>
    <td>71.7</td>
    <td>74.8</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ViT-G/14</td>
    <td>dinov2</td>
    <td>62.3</td>
    <td>69.9</td>
    <td>73.6</td>
    <td>-</td>
  </tr>
</table>


### Usage

#### Example on how to Evaluate dino with the Hummingbird (Dense NN Retrieval) Evaluation  on Pascal VOC

```python
import torch
from src.hbird_eval import hbird_evaluation
# Parameters for the model dino
device = 'cuda'
input_size = 224
batch_size = 64
patch_size = 16
embed_dim = 384
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

# Define the function to extract features from the model
# Input to the function is the model and the images
# Output of the function is the features extracted from the model 
# and optionally the attention maps
fn = lambda model, imgs: (model.get_intermediate_layers(imgs)[0][:, 1:], None)


# Evaluate the model using the Full In-Context Learning Hummingbird  
# or Dense k-NN Retrieval Evaluation on the Pascal VOC Dataset
hbird_miou = hbird_evaluation(model.to(device), 
        d_model=embed_dim,          # size of the embedding feature vectors of patches
        patch_size=patch_size, 
        batch_size = batch_size, 
        input_size=input_size,             
        augmentation_epoch=1,       # how many iterations of augmentations to use on top of 
                                    # the training dataset in order to generate the memory
        device=device,              
        return_knn_details=False,   # whether to return additional NNs details
        n_neighbours=30,            # the number of neighbors to fetch per image patch
        nn_method='<nn method>',    # options: faiss or scann as the k-nn library to be used, scann uses cpu, faiss gpu
        nn_params=None,             # Other parameters to be used for the k-NN operator
        ftr_extr_fn=fn,             # function that extracts image patch features with 
                                    # a vision encoder
        dataset_name='voc',         # the name of the dataset to use, 
                                    # currently only Pascal VOC is included.
        data_dir='<the path to the Pascal VOC Dataset>',    # path to the dataset 
                                                            # to use for evaluation
        memory_size=None,           # How much you want to limit your datasetNone if to be left unbounded
        train_fs_path=None,         # The path to the file with the subset of filenames for training
        val_fs_path=None,           # The path to the file with the subset of filenames for validation
    )           
                                    
print('Dense NN Ret - miou score:', hbird_miou) 

```

#### Example on how to Evaluate dinov2 with Dense NN Retrieval on Pascal VOC

```python
import torch
from src.hbird_eval import hbird_evaluation
# Parameters for the model dino
device = 'cuda'
input_size = 224
batch_size = 256
patch_size = 14
embed_dim = 384
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# Define the function to extract features from the model
# Input to the function is the model and the images
# Output of the function is the features extracted from the model 
# and optionally the attention maps
fn = lambda model, imgs: (model.forward_features(imgs)['x_norm_patchtokens'], None)


# Evaluate the model using the Full In-Context Learning Hummingbird  
# or Dense k-NN Retrieval Evaluation on the Pascal VOC Dataset
hbird_miou = hbird_evaluation(model.to(device), 
        d_model=embed_dim,          # size of the embedding feature vectors of patches
        patch_size=patch_size, 
        batch_size = batch_size, 
        input_size=input_size,             
        augmentation_epoch=1,       # how many iterations of augmentations to use on top of 
                                    # the training dataset in order to generate the memory
        device=device,              
        return_knn_details=False,   # whether to return additional NNs details
        n_neighbours=30,            # the number of neighbors to fetch per image patch
        nn_method='<nn method>',    # options: faiss or scann as the k-nn library to be used, scann uses cpu, faiss gpu
        nn_params=None,             # Other parameters to be used for the k-NN operator
        ftr_extr_fn=fn,             # function that extracts image patch features with 
                                    # a vision encoder
        dataset_name='voc',         # the name of the dataset to use, 
                                    # currently only Pascal VOC is included.
        data_dir='<the path to the Pascal VOC Dataset>',    # path to the dataset 
                                                            # to use for evaluation
        memory_size=None,           # How much you want to limit your datasetNone if to be left unbounded
        train_fs_path=None,         # The path to the file with the subset of filenames for training
        val_fs_path=None,           # The path to the file with the subset of filenames for validation
    )   
print('Dense NN Ret - miou score:', hbird_miou) 

```

#### Ready to use script

We also provide a ready to use Python script to run evaluations using DINO backbones. For example, to evaluate a ViT S/16 on the whole Pascal VOC dataset using a memory bank of size 1024*10<sup>2</sup> you can run the following command

```sh
python eval.py                  \
    --seed 42                   \
    --batch-size 64             \
    --input-size 512            \
    --patch-size 16             \
    --memory-size 102400        \
    --embeddings-size 384       \
    --data-dir VOCSegmentation  \
    --model dino_vits16
```

###  Setup
This is the section describing what is required to execute the Dense NN Retrieval Evaluation.
Installation instructions can be found to the [Installation Guide](./INSTALLATION.md).

* If you want to install the library to have access everywhere when using a python environment, then you can do so:
```bash
cd open-hummingbird-eval
pip install . # or for editing and using: `pip install -e .`
```

#### Dataset Setup
We now have 5 available datasets:
* `ade20k`
* `voc`
* `coco-thing`
* `coco-stuff`
* `cityscapes`

And you can now select a subset of the training dataset by including a suffix `*<fraction>` next to the dataset name. For example, to evaluate on a random `0.1` fraction of the Pascal VOC, you can specify `dataset_name="voc*0.1"` in the hbird_evaluation evaluation method above.

Please refer to the [Dataset Guide](./DATASET.md) to see the full structure of how each dataset folder should look like.

We also provide file sets in the [file_sets](./file_sets/) folder that specify subsets of the original dataset. For example in the folder [./file_sets/voc/1_div_8/](./file_sets/voc/1_div_8/) there are 5 different subsets of the Pascal VOC training dataset that are a 1/8 fraction of the original dataset, keeping the same distribution of labels as the original dataset ( for the 1_div_128 that would be 1/128 fraction etc.).

### Examples
Basic examples on how to download any of our dataset versions and evaluate a vision encoder with our implementation of the Hummingbird evaluation can be found at the [examples folder](./examples/).

You can also open it in google colab:
#### Example with using scann library
<a href="https://githubtocolab.com/vpariza/open-hummingbird-eval/blob/main/examples/hbird_eval_example_scann.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Example with using faiss-gpu library
<a href="https://githubtocolab.com/vpariza/open-hummingbird-eval/blob/main/examples/hbird_eval_example_faiss_gpu.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Upcoming/Future Features
Stay tuned with our work because we will bring more support and extensions of our implementation for extra features.

| Feature | Description |
| --- | --- |
| `NYUv2` | Support for Depth Estimation with this code for **NYUv2** |

## Contributors

| n  | Username |
| ------------- | ------------- |
| 1  | [@vpariza](https://github.com/vpariza)  |
| 2  | [@Smsd75](https://github.com/Smsd75) |
| 3  | [@yukimasano](https://github.com/yukimasano) |

## Citations
If you find this repo helpful, please consider citing these works:

The original paper:
```
@inproceedings{
      balazevic2023towards,
      title={Towards In-context Scene Understanding},
      author={Ivana Balazevic and David Steiner and Nikhil Parthasarathy and Relja Arandjelovic and Olivier J Henaff},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=FasIQqsJhe}
}
```

Our work and repository:
```
@misc{pariza2024hbird,
      author = {Pariza, Valentinos and Salehi, Mohammadreza and Asano, Yuki},
      month = {4},
      title = {Hummingbird Evaluation for vision encoders},
      url = {https://github.com/vpariza/open-hummingbird-eval},
      year = {2024}
}
```

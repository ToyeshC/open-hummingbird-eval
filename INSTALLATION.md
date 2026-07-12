Installation instructions
## Step 1: Create a python environment conda (+ install basic libraries)
```bash
conda create -n "hbird" python=3.12 ipython pylint ipykernel 
```

## Step 2: Activate the conda environment
```bash
conda activate hbird
```
or 
```bash
source activate hbird
```

## Step 3: Update pip
```bash
python -m pip install --upgrade pip
```

## Step 4: Install the libraries (Multiple Examples) 
### Use faiss-gpu for Nerarest Neighbor Retrieval
#### Approach 1: Use CUDA 11.8
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

conda install -c pytorch -c nvidia -c rapidsai -c conda-forge libnvjitlink faiss-gpu-cuvs=1.11.0

pip install lightning==2.5.5
pip install tqdm==4.67.1
pip install numpy==1.26.4
pip install scipy==1.11.4
```
You could also find the singularity definition of this approach at [hbird_cuda11_8.def](./singularity_defs/hbird_cuda11_8.def)

#### Approach 2: Use CUDA 12.1
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install faiss-gpu-cu12

pip install lightning==2.5.5
pip install tqdm==4.67.1
pip install numpy==1.26.4
pip install scipy==1.11.4
```
You could also find the singularity definition of this approach at [hbird_cuda12_1.def](./singularity_defs/hbird_cuda12_1.def)

#### Approach 3: Use CUDA 12.6
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu126

conda install -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs "cuda-version>=12.6,<12.7" "numpy>=2.0,<2.3" "scipy>=1.14,<1.16"

pip install lightning==2.5.5
pip install tqdm==4.67.1
```
You could also find the singularity definition of this approach at [hbird_cuda12_6.def](./singularity_defs/hbird_cuda12_6.def)

### Use scann (on cpu)for Nerarest Neighbor Retrieval
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu126

pip install faiss-cpu
pip install scann

pip install lightning==2.5.5
pip install tqdm==4.67.1
```
You could also find the singularity definition of this approach at [hbird_cpu.def](./singularity_defs/hbird_cpu.def)

## Step 5: Encoder-specific dependencies (optional)

The base install above covers DINO and DINOv2, which load through `torch.hub` and need no
extra packages. The other supported encoders need additional libraries:

- **Hugging Face encoders** (CLIP, SigLIP2, RADIO, DINOv3):
  ```bash
  pip install "transformers>=4.34,<5" huggingface-hub
  ```
  RADIO additionally needs:
  ```bash
  pip install timm open_clip_torch einops
  ```
- **TIPS** (loaded from a local checkout):
  ```bash
  pip install tensorflow_text mediapy jax jaxlib scikit-learn
  git clone https://github.com/google-deepmind/tips
  export PYTHONPATH="$PYTHONPATH:/path/to/tips"   # then run its download_checkpoints.sh
  ```
- **Optional performance boost:** `pip install xformers`

`transformers` must stay `<5` with torch 2.2.2 (transformers 5.x requires torch >= 2.4).

### Verified environment

The MVImgNet 3D-evaluation results were produced with **torch 2.2.2+cu121, transformers
4.51.3, faiss-gpu-cu12, numpy 1.26.4** — Approach 2 (CUDA 12.1) above plus the Hugging
Face dependencies.

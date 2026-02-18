<div align="center">
<h1>PatchRefinerV2 </h1>
<h3>Fast and Lightweight Real-Domain High-Resolution <br> Metric Depth Estimation</h3>

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2501.01121) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

<a href="https://zhyever.github.io/">Zhenyu Li</a>, <a href="https://www.linkedin.com/in/wenqing-cui-a2434431a/?originalSubdomain=sa">Wenqing Cui</a>, <a href="https://shariqfarooq123.github.io/">Shariq Farooq Bhat</a>, <a href="https://peterwonka.net/">Peter Wonka</a>. 
<br>KAUST

</div>

## ✨ **NEWS**
- 2026-02-17: PRV2 is accepted to ICLR 2026. Finish releasing everything.
- 2025-01-05: Release codes. Pretrained models are coming soon.

---

## Installation

```bash
conda env create -n patchrefinerv2 --file environment.yml
conda activate patchrefinerv2
```

---

## Data Preparation

### UnrealStereo4K (U4K)

Download the dataset and place (or symlink) it under `data/u4k/`. The expected directory structure is:

```
data/u4k/
├── splits/
│   ├── val.txt
│   ├── test_in.txt
│   └── test_out.txt
└── ... (images and depth maps)
```

The split files are provided in `splits/u4k/`. You can symlink them:

```bash
ln -s $(pwd)/splits/u4k $(pwd)/data/u4k/splits
```

### Cityscapes

Download the Cityscapes dataset and place (or symlink) it under `data/cityscapes/`. The split file `data/cityscapes/val.txt` listing relative image paths is required, which is also provided in `splits/cityscapes/`

---

## Checkpoints

Download the pretrained checkpoints from [here](https://drive.google.com/drive/folders/122ftVbSHBOgGcVv6fXY3rbpM69pID_82?usp=sharing):

| Model | Dataset | Config |
|-------|---------|--------|
| PatchRefinerV2-U4K | UnrealStereo4K | `configs/patchrefinerv2_u4k.yaml` |
| PatchRefinerV2-CS | Cityscapes | `configs/patchrefinerv2_cityscapes.yaml` |

Place checkpoints anywhere and pass the path via `--ckp-path`.

---

## Inference

All inference is launched via `tools/dist_test.sh`, which automatically sets `PYTHONPATH` and spawns the distributed launcher.

```bash
bash tools/dist_test.sh <CONFIG> <NUM_GPUS> --ckp-path <CHECKPOINT> [OPTIONS]
```

### Benchmark Evaluation

#### UnrealStereo4K

```bash
bash tools/dist_test.sh configs/patchrefinerv2_u4k.yaml 1 \
    --ckp-path /path/to/u4k_checkpoint.pth \
    --save \
    --work-dir ./output/u4k
```

#### Cityscapes

```bash
bash tools/dist_test.sh configs/patchrefinerv2_cityscapes.yaml 1 \
    --ckp-path /path/to/cs_checkpoint.pth \
    --image-raw-shape 1024 2048 \
    --patch-split-num 4 4 \
    --cai-mode m1 \
    --save \
    --work-dir ./output/cityscapes
```

### Custom Image Inference

Provide a directory of images via `--cfg-options` and use `--test-type general`:

```bash
bash tools/dist_test.sh configs/patchrefinerv2_u4k.yaml 1 \
    --ckp-path /path/to/u4k_checkpoint.pth \
    --test-type general \
    --cfg-options general_dataloader.dataset.rgb_image_dir='/path/to/images' \
    --cai-mode r64 \
    --image-raw-shape 2160 3840 \
    --patch-split-num 4 4 \
    --save \
    --work-dir ./output/custom
```

---

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ckp-path` | Path to the `.pth` checkpoint file | `''` |
| `--test-type` | `normal` for benchmark datasets, `general` for custom images | `normal` |
| `--cai-mode` | Tiling mode: `m1` (non-overlapping grid), `m2` (shifted tiles), `rN` (N random patches, e.g. `r64`) | `m1` |
| `--image-raw-shape H W` | Full resolution of the input images (height width) | `2160 3840` |
| `--patch-split-num R C` | Number of patch rows and columns | `4 4` |
| `--process-num` | Batch size for patch inference (reduce if OOM) | `4` |
| `--save` | Save predicted depth maps and colorized visualizations | off |
| `--work-dir` | Output directory for logs and predictions | derived from `--ckp-path` |
| `--cfg-options` | Override any config value, e.g. `key=value` | — |
| `--gray-scale` | Use grayscale colormap for visualizations | off |

### Tiling Modes

- **`m1`**: Non-overlapping grid. Fast and memory-efficient.
- **`m2`**: Shifted-grid (overlapping) tiles. Better boundary handling.
- **`rN`**: N random patches sampled from the m2 set (e.g. `r64`). Good balance of coverage and speed.

### Patch Split Constraints

The image dimensions must satisfy:
- `height` divisible by `2 × patch_split_num[0]`
- `width` divisible by `2 × patch_split_num[1]`

For example, with `--image-raw-shape 2160 3840 --patch-split-num 4 4`:
- 2160 / (2×4) = 270 ✓
- 3840 / (2×4) = 480 ✓
- Check [here](https://github.com/zhyever/PatchFusion/blob/main/docs/user_infer.md#user-inference) for more details.

---

## Multi-GPU Inference

Pass the number of GPUs as the second argument to `dist_test.sh`:

```bash
bash tools/dist_test.sh configs/patchrefinerv2_u4k.yaml 4 \
    --ckp-path /path/to/checkpoint.pth \
    --save \
    --work-dir ./output/u4k
```

---


## Citation
If you find our work useful for your research, please consider citing the paper
```
@article{li2025patchrefinerv2,
    title={PatchRefiner V2: Fast and Lightweight Real-Domain High-Resolution Metric Depth Estimation}, 
    author={Li, Zhenyu and Cui, Wenqing and Bhat, Shariq Farooq and Wonka, Peter},
    journal={arXiv preprint arXiv:2501.01121},
    year={2025}
}
```

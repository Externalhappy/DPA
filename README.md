# DPA: Dual Prototypes Alignment for Unsupervised Adaptation of Vision-Language Models

This is the PyTorch code of the [DPA](https://arxiv.org/abs/2408.08855) paper.

## Dataset Setup

Dataset paths are stored in `dataset_catalog.json`, which need to be modified to local paths. Please refer to the scripts from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets) to download and prepare. CLIP's labels are stored in `classes.json`.

## Training

Run the following command:

```bash
python train.py --dataset [name_of_dataset]
```

## Citation

```bibtex
@InProceedings{Ali_2025_WACV,
    author    = {Ali, Eman and Silva, Sathira and Khan, Muhammad Haris},
    title     = {DPA: Dual Prototypes Alignment for Unsupervised Adaptation of Vision-Language Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {6083-6093}
}

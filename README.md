# PIR-CLIP: Remote Sensing Image-Text Retrieval with Prior Instruction Representation Learning
By [Jiancheng Pan](https://scholar.google.com/citations?user=nRPD3tAAAAAJ&hl=en&oi=ao), Muyuan Ma, Qing Ma, [Cong Bai](https://scholar.google.com/citations?hl=zh-CN&user=XGZ4UZgAAAAJ&view_op=list_works&sortby=pubdate) and [Xiaomeng Huang](https://scholar.google.com/citations?hl=zh-CN&user=yH9OkqYAAAAJ&view_op=list_works&sortby=pubdate).

This repo is the official implementation of "[PIR-CLIP: Remote Sensing Image-Text Retrieval with Prior Instruction Representation Learning]()".

- [PIR-CLIP: Remote Sensing Image-Text Retrieval with Prior Instruction Representation Learning](#pir-clip-remote-sensing-image-text-retrieval-with-prior-instruction-representation-learning)
  - [ℹ️ Introduction](#ℹ️-introduction)
  - [🎯 Implementation](#-implementation)
    - [Environments](#environments)
    - [Train](#train)
    - [Retrieval](#retrieval)
  - [🌎 Datasets](#-datasets)
  - [📝 Citation](#-citation)

## ℹ️ Introduction
This paper introduces a Prior Instruction Representation (PIR) learning paradigm that draws on prior knowledge to instruct adaptive learning of vision and text representations. Firstly, a Prior Instruction Remote Sensing Image-Text Retrieval (PIR-ITR) framework is designed to address semantic noise issues in vision-language understanding tasks. Specifically, we introduce two Progressive Attention Encoder (PAE) structures: Spatial-PAE and Temporal-PAE, aimed at modeling long-range dependencies to enhance the representation of key features. In vision representation, Vision Instruction Representation (VIR) based on Spatial-PAE utilizes the prior-guided knowledge of the remote sensing scene recognition by building a belief matrix to select key features for reducing the impact of semantic noise. In text representation, Language Cycle Attention (LCA) based on Temporal-PAE uses the previous time step to cyclically activate the current time step to enhance text representation capability. A cluster-wise affiliation loss is proposed to constrain the inter-classes and to reduce the semantic confusion zones in the common subspace. \textcolor{black}{Based on PIR, we propose PIR-CLIP, a domain-specific CLIP-based framework for RSITR, to further improve open-domain retrieval performance.} Comprehensive experiments demonstrate that using prior knowledge instruction could enhance vision and text representations and could outperform the state-of-the-art methods on two benchmark datasets, RSICD and RSITMD.
![pipline](assets/pipline.png)

## 🎯 Implementation
### Environments
base on `open_clip` environments, you can click here [open_clip](https://github.com/mlfoundations/open_clip).

### Train
If using Affiliation loss, add `is_aff_loss` where the label information is obtained by `image_name` from datasets. For example, we can train PIR-CLIP using the follow commad:
```
python -m training.main \
    --save-frequency 1 \
    --report-to tensorboard \
    --train-data="path/to/webdataset/tar" \
    --dataset-resampled \
    --train-num-samples num_dataset \
    --dataset-type webdataset \
    --warmup 10000 \
    --batch-size=512\
    --precision amp \
    --lr=1e-5 \
    --wd=0.5 \
    --epochs=20 \
    --workers=4 \
    --model=PIR \
    --is_aff_loss
```
or parallel training as
```
torchrun --nproc_per_node 2 \
    --rdzv_endpoint=$HOSTE_NODE_ADDR \
    -m training.main \
    --save-frequency 1 \
    ...
```
### Retrieval
Retrieval evaluation on [CLIP Benchmark](https://github.com/ChenDelong1999/RemoteCLIP) and checkpoints can download from here: [Baidu Disk]().
```
python retrieval.py \
    --model-name "PIR" \
    --retrieval-images-dir "path/to/images" \
    --retrieval-json-dir "path/to/dataset.json" \
    --remoteclip-path "./checkpoints/PIR-CLIP_RSITMD.pt"
```
## 🌎 Datasets

All experiments are based on [RSITMD](https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD), [RSICD](https://github.com/201528014227051/RSICD_optimal) datasets and pre-training dataset [RS5M](https://github.com/om-ai-lab/RS5M).

## 📝 Citation

If you find this code useful for your work or use it in your project, please cite our paper as:

```
@inproceedings{pan2023prior,
  title={A Prior Instruction Representation Framework for Remote Sensing Image-text Retrieval},
  author={Pan, Jiancheng and Ma, Qing and Bai, Cong},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={611--620},
  year={2023}
}
```
# PIR-CLIP: Remote Sensing Image-Text Retrieval with Prior Instruction Representation Learning
By [Jiancheng Pan](https://scholar.google.com/citations?user=nRPD3tAAAAAJ&hl=en&oi=ao), Muyuan Ma, Qing Ma, [Cong Bai](https://scholar.google.com/citations?hl=zh-CN&user=XGZ4UZgAAAAJ&view_op=list_works&sortby=pubdate) and Xiaomeng Huang.

This repo is the official implementation of "[PIR-CLIP: Remote Sensing Image-Text Retrieval with Prior Instruction Representation Learning]()".


## ℹ️ Introduction
This paper presents a prior instruction representation (PIR) learning paradigm that draws on prior knowledge to instruct adaptive learning of vision and text representations. Firstly, a prior instruction remote sensing image-text retrieval framework (PIR-ITR) , aimed at remote sensing vision-language understanding tasks to solve the semantic noise problem. Concretely, two progressive attention encoder (PAE) structures, Spatial-PAE and Temporal-PAE, are proposed to perform long-range dependency modeling to enhance key feature representation. In vision representation, Vision Instruction Representation (VIR) based on Spatial-PAE exploits the prior-guided knowledge of the remote sensing scene recognition by building a belief matrix to select key features for reducing the impact of semantic noise. In text representation, Language Cycle Attention (LCA) based on Temporal-PAE uses the previous time step to cyclically activate the current time step to enhance text representation capability. A cluster-wise affiliation loss is proposed to constrain the inter-classes and to reduce the semantic confusion zones in the common subspace. Based on PIR, we propose PIR-CLIP, a domain-specific CLIP-based framework for RSITR, to fill in the gaps of comparisons with open-domain retrieval methods on the remote sensing domain Comprehensive experiments demonstrate that using prior knowledge instruction could enhance vision and text representations and could outperform the state-of-the-art methods on two benchmark datasets, RSICD and RSITMD.
![pipline](assets/pipline.png)

### Environments
base on `open_clip` environments

### Retrieval
checkpoints can download from here: [Baidu Disk]()
Retrieval Evaluation on [clip_benchmark](https://github.com/ChenDelong1999/RemoteCLIP).
```
python retrieval.py \
    --model-name "PIR" \
    --retrieval-images-dir "path/to/images" \
    --retrieval-json-dir "path/to/dataset.json" \
    --remoteclip-path "./checkpoints/PIR-CLIP_RSITMD.pt"
```

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
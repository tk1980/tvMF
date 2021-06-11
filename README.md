# t-vMF Similarity

The Pytorch implementation for the CVPR2021 paper of "[t-vMF Similarity for Regularizing In-Class Feature Distribution](https://staff.aist.go.jp/takumi.kobayashi/publication/2021/CVPR2021.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2021cvpr,
  title={t-vMF Similarity for Regularizing In-Class Feature Distribution},
  author={Takumi Kobayashi},
  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

## Introduction

This work proposes a method to effectively impose regularization on feature representation learning.
By focusing on the angle between a feature and a classifier which is embedded in cosine similarity at the classification layer, we formulate a novel similarity beyond the cosine based on von Mises-Fisher distribution of directional statistics.
In contrast to the cosine similarity, our similarity is compact while having heavy tail, which contributes to regularizing intra-class feature distribution to improve generalization performance.
The method works in the classification loss in stead of cosine similarity without adding regularization loss, to improve feature representation trained on ill-conditioned datasets, such as imbalanced, small-scaled and noisy-labeled datasets.
It can also improve performance on large-scale healthy dataset such as ImageNet.
For more detail, please refer to our [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2021/CVPR2021.pdf).

<img width=400 src="https://user-images.githubusercontent.com/53114307/121633581-04ea6800-cabe-11eb-905b-a39c0a88da8e.png">


## Usage

### Training
The algorithm is simply implemented in the class `tvMFLoss` at `models/Loss.py`; the core is found in line 36 of `models/Loss.py`.

For example, the ResNet-10 equipped with the t-vMF similarity is trained from scratch on ImageNet-LT dataset by
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_train.py --dataset imagenetlt  --data ./datasets/imagenetlt --net-config ResNet10Feature  --loss-config tvMFLoss_k16 --out-dir ./results/imagenetlt/ResNet10Feature/tvMFLoss_k16/ --workers 12 --seed 0 

CUDA_VISIBLE_DEVICES=0,1 python main_finetune.py --dataset imagenetlt  --data ./datasets/imagenetlt --net-config ResNet10Feature_finetune  --loss-config CosLoss --model-file ./results/imagenetlt/ResNet10Feature/tvMFLoss_k16//model_best.pth.tar --out-dir ./results/imagenetlt/ResNet10Feature/tvMFLoss_k16/finetune --workers 12 --seed 0
```

Note that the ImageNet-LT dataset must be downloaded at `./datasets/imagenetlt/` before the training and we follow the imbalance-aware training procedure presented in [1]; the standard training is performed by `main_train.py` and then fine-tuning via class-imbalance aware sampling is done by `main_finetune.py`.

Or, in short, you can simply run the script by
```bash
sh do_imbalance.sh 0,1 imagenetlt ResNet10Feature tvMFLoss_k16
```

## Results

#### ImageNet

| Method  | ImageNet-LT | iNaturalist-2018 |
|---|---|---|
| t-vMF ({\kappa=16}) | 57.30   | 28.92|
| t-vMF ({\kappa=64}) | 56.31   | 29.69|
| Others |  58.2 [1] | 32.00 [2] |


## References

[1] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng and Yannis Kalantidis. "DECOUPLING REPRESENTATION AND CLASSIFIER FOR LONG-TAILED RECOGNITION." In ICLR, 2020.

[2] Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga and Tengyu Ma. "Learning imbalanced datasets with label-distribution-aware margin loss." In NeurIPS, 2019.


## Contact
takumi.kobayashi (At) aist.go.jp


## Acknowledgement
The class-wise sampler `utils/ClassAwareSampler.py` is from the [Classifier-Balancing](https://github.com/facebookresearch/classifier-balancing).
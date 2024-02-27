# vision_transformer_pytorch
简化ViT代码，仅ViT-B-16模型，便于学习

## 指标展示
|Model| dataset | net_size | Top1 |
|-----|------|------|-----|
| ViT-B-16(paper) | CIFAR10 | 224x224 |	0.9900 |
| **ViT-B-16(ours)** | CIFAR10 | 224x224 |	**0.9878** |

## 使用说明
### 要求
> Python >= 3.6 \
> PyTorch >= 1.7
### 数据集下载
```shell script
cd data/
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz  
tar -xzvf cifar-10-python.tar.gz 
```
### 预训练模型下载
[vit_b_16_224.pth(提取码8888)](https://pan.baidu.com/s/1WXfNyW3fahlQpM2LAERbfQ)
### 训练
```shell script
python train.py
```
## 参考
https://github.com/google-research/vision_transformer  
https://github.com/jeonsworld/ViT-pytorch  

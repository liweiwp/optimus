# 任务4 基于视觉传感器的3D目标检测
概述：多视觉传感器的3D目标检测算法主要应用于汽车智能驾驶，基于车载多摄像头的输入，完成车辆周围多目标的3D位置的检测，并能适应城市工况下的复杂场景。
且指标性能接近使用激光雷达的检测水平。

指标：**NDS** > 0.569, **mAP** > 0.481

***

## 1 学习
> - 从零开始学习机器学习，深度学习知识
> - 掌握软件工具NumPy, PyTorch，
> - 阅读论文学习算法思想，并复现论文算法
### 课程
课程视频在B站有搬运
* [CS229 Machine Learning](https://cs229.stanford.edu/) 机器学习
* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) 深度学习
* [CS231n Deep Learning for Computer Vision](http://cs231n.stanford.edu/) 计算机视觉
* [Self-Driving Cars Specialization](https://www.coursera.org/specializations/self-driving-cars) 自动驾驶汽车
### 软件
* [NumPy](https://numpy.org/) 数值计算
* [PyTorch](https://pytorch.org/) GPU加速的数值计算，神经网络
* [mmdet](https://github.com/open-mmlab/mmdetection) 目标检测
* [mmdet3d](https://github.com/open-mmlab/mmdetection3d) 3D目标检测
### 论文
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) ResNet
* [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144) FPN
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762) Transformer
* [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) DETR
* [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://arxiv.org/abs/2110.06922) DETR3D
* [PETR: Position Embedding Transformation for Multi-View 3D Object Detection](https://arxiv.org/abs/2203.05625v3) PETR
* [PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images](https://arxiv.org/abs/2206.01256) PETRv2
* [BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270v2) BEVFormer

## 2 数据集
> 探索数据，数据预处理
### nuScenes 数据集

### KITTI 数据集

## 3 算法
通过复现基本的算法，掌握算法的原理，并提出自己的改进。
||**NDS**|**mAP**|
|----|----|----|
|DETR3D|0.479|0.412|
|PETR|0.481|0.434|
|BEVFormer|0.569|0.481|
|PETRv2|0.582|0.490|
### DETR
Official Code [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)
### DETR3D
Official Code [https://github.com/wangyueft/detr3d](https://github.com/wangyueft/detr3d)
### BEVFormer
Official Code [https://github.com/fundamentalvision/BEVFormer](https://github.com/fundamentalvision/BEVFormer)
### PETR, PETRv2
Official Code [https://github.com/megvii-research/PETR](https://github.com/megvii-research/PETR)

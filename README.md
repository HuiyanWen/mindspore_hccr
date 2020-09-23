# 使用MindSpore训练CNN手写汉字识别

## 完整效果
![t](https://github.com/HuiyanWen/Atlas200_HCCR/blob/master/tx6or-k8jom.gif)

<div><img width="300" height="200" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/hanzi28.jpg"/></div>
## 整体训练流程
<div><img width="300" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E6%95%B4%E4%BD%93%E6%B5%81%E7%A8%8B.png"/></div>

## 代码说明
```
code
└── data_preprocessing
	├── gnt2gnt.cpp                     # 数据预处理
	└── gnt2tf.py                       # 制作tfrecord数据集以供mindspore训练
└── train&eval
	├── config.py                       # 配置文件
	├── dataset.py                      # 数据载入及预处理
	├── eval.py                         # 批量验证测试集精度&导出pb模型
	├── lr_generator.py                 # 学习率设置
	├── train_with_eva.py               # 训练并验证精度
	├── inference.py              	    # 推理单张图片
	└── resnet.py                       # resnet18
```
## 图像预处理
该部分可参考[手写汉字拍照识别系统](https://www.huaweicloud.com/ascend/apps/applicationDetails/812608665)，处理方法类似
## 网络模型ResNet-18
该部分使用标准resnet论文中的18层结构。由于手写汉字特征较为简单且三通道意义不大，因此将输入改为了112 * 112 * 1。

<div><img width="600" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/resnet.png"/></div>
## 创建作业开始云端训练
**选择华为云服务中的ModelArts；选择训练管理中的训练作业模块，点击“创建”。**

<div><img width="600" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%88%9B%E5%BB%BA%E4%BD%9C%E4%B8%9A%E5%BC%80%E5%A7%8B%E4%BA%91%E7%AB%AF%E8%AE%AD%E7%BB%83.png"/></div>
**<br>在常用框架模块下进行设置，包括对代码目录、启动文件、数据存储位置和单卡多卡模式等。**

<div><img width="600" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%88%9B%E5%BB%BA%E4%BD%9C%E4%B8%9A%E5%BC%80%E5%A7%8B%E4%BA%91%E7%AB%AF%E8%AE%AD%E7%BB%832.png"/></div>
**<br>在训练作业中选择一个任务进入，选择日志模块进行查看。若在本地配置了MindSpore，还可通过MindInsight可视化训练过程。**

<div><img width="600" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%87%202.png"/></div>
## 训练结果
mindspore的底层加速确实不错，同等参数训练比tensorflow要快一些。

<div><img width="450" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%873.png"/></div>
## 模型导出
网络训练完成后，可进一步导出为GEIR或ONNX格式的PB模型，以便后续部署到Atlas或其它平台上进行推理。

<div><img width="500" height="300" src="https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%87%204.png"/></div>

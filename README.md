# mindspore_hccr
使用mindspore训练手写汉字，并部署到Atlas200dk上实现边缘推理。

## 完整效果
![t](https://github.com/HuiyanWen/Atlas200_HCCR/blob/master/tx6or-k8jom.gif)

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/hanzi28.jpg)
## 整体训练流程
![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E6%95%B4%E4%BD%93%E6%B5%81%E7%A8%8B.png)
## 数据集制作文件说明
`gnt2gnt.cpp:数据预处理
`
 <br>
` gnt2tf.py:制作tfrecord数据集以供mindspore训练
`
## 图像预处理
该部分可参考[手写汉字拍照识别系统](https://www.huaweicloud.com/ascend/apps/applicationDetails/812608665)，处理方法类似
## 网络模型ResNet-18
该部分使用标准resnet论文中的18层结构。由于手写汉字特征较为简单且三通道意义不大，因此将输入改为了112 * 112 * 1。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/resnet.png)
## 创建OBS
选择华为云服务中的对象存储服务OBS；点击“创建桶”，根据需要选择不同计费标准。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%87%205.png)
## 上传数据给OBS及授权给ModelArts使用
该部分可参考[官网教程](https://bbs.huaweicloud.com/videos/101366)。
## 创建作业开始云端训练
选择华为云服务中的ModelArts；选择训练管理中的训练作业模块，点击“创建”。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%88%9B%E5%BB%BA%E4%BD%9C%E4%B8%9A%E5%BC%80%E5%A7%8B%E4%BA%91%E7%AB%AF%E8%AE%AD%E7%BB%83.png)
<br>在常用框架模块下进行设置，包括对代码目录、启动文件、数据存储位置和单卡多卡模式等。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%88%9B%E5%BB%BA%E4%BD%9C%E4%B8%9A%E5%BC%80%E5%A7%8B%E4%BA%91%E7%AB%AF%E8%AE%AD%E7%BB%832.png)
<br>在训练作业中选择一个任务进入，选择日志模块进行查看。若在本地配置了MindSpore，还可通过MindInsight可视化训练过程。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%87%201.png)
![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%87%202.png)
## 训练结果
mindspore的底层加速确实不错，同等参数训练比tensorflow要快一些。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%873.png)
## 模型导出
网络训练完成后，可进一步导出为GEIR或ONNX格式的PB模型，以便后续部署到Atlas或其它平台上进行推理。

![t](https://github.com/HuiyanWen/mindspore_hccr/blob/master/pic/%E5%9B%BE%E7%89%87%204.png)


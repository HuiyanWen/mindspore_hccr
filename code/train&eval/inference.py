import os
import argparse
import mindspore.nn as nn
import numpy as np
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
#from src.dataset import create_dataset
from dataset import create_dataset
from mindspore.nn.optim.momentum import Momentum
import mindspore.train.serialization as serialization
from config import config
from lr_generator import get_lr
import mindspore
from mindspore import Tensor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
#from src.lenet import LeNet5
import moxing as mox
from PIL import Image
from resnet import resnet18

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="s3://hithcd/rgb/pic/",
                        help='path where the dataset is saved')

    parser.add_argument('--ckpt_path', type=str, default="obs://hithcd/MA-hw_project_resnet18-05-30-19/output/V0408/", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    local_data_url = '/cache/data'
    local_output_url = '/cache/ckpt'
    mox.file.copy_parallel(args.data_path, local_data_url)
    mox.file.copy_parallel(args.ckpt_path, local_output_url)

    net = resnet18(class_num=config.class_num)

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(os.path.join(local_output_url, 'resnet-50_1759.ckpt'), net=net)
    load_param_into_net(net, param_dict)
    im = np.asarray(Image.open(os.path.join(local_data_url, '490.png')).convert('L'))
    im = 255-im
    im = im/255.0
    input = im.reshape((1, 1, 112, 112))
    input_tensor = Tensor(input, mindspore.float32)
    acc = net(input_tensor)
    acc = acc.asnumpy()
    preds = np.argmax(acc, axis=1)
    mox.file.copy_parallel(local_output_url, args.train_url)
    print("Predict label:{0}, acc={1}".format(preds[0], acc[0][preds[0]]))
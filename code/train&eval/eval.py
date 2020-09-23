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
from mindspore import Tensor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
#from src.lenet import LeNet5
import moxing as mox
from resnet import resnet18

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="s3://hithcd/gray/",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="obs://hithcd/MA-hw_project_resnet18-05-30-19/output/V0429/", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    local_data_url = '/cache/data'
    local_output_url = '/cache/ckpt'
    mox.file.copy_parallel(args.data_path, local_data_url)
    mox.file.copy_parallel(args.ckpt_path, local_output_url)

    ds_eval = create_dataset(dataset_path=os.path.join(local_data_url, 'test'), do_train=False,
                             batch_size=config.batch_size)
    epoch_size = config.epoch_size
    net = resnet18(class_num=config.class_num)
    #print("#####:", net.shape)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    step_size = ds_eval.get_dataset_size()
    lr = Tensor(get_lr(global_step=0, lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                       lr_decay_mode='steps'))

    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)

    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(os.path.join(local_output_url, 'resnet-20_1759.ckpt'))
    load_param_into_net(net, param_dict)
    input = np.random.uniform(0.0, 1.0, size=[1, 1, config.image_height, config.image_width]).astype(
        np.float32)
    serialization.export(net, Tensor(input), file_name='/cache/ckpt/resnet_gray_softmax.pb', file_format='GEIR')
    mox.file.copy_parallel(local_output_url, args.train_url)
    # ds_eval = create_dataset(os.path.join(args.data_path, "test"),
    #                          config.batch_size,
    #                          1)
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train_imagenet."""
import os
import argparse
import random
import numpy as np
from dataset import create_dataset
from lr_generator import get_lr
from config import config
from mindspore import context
from mindspore import Tensor
from resnet import resnet18
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindspore.train.model import Model, ParallelMode

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset.engine as de
from mindspore.communication.management import init
import moxing as mox
import mindspore.train.serialization as serialization
random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
# parser.add_argument('--run_distribute', type=bool, default=False, help='Run distribute')
# parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
# parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
args_opt = parser.parse_args()

device_id = int(os.getenv('RANK_ID'))
device_num = int(os.getenv('RANK_SIZE'))

local_data_url = '/cache/data'
local_output_url = '/cache/ckpt'

from mindspore.ops import operations as P
from mindspore import Tensor
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        #self.fc1 = nn.Dense(10000, 120)
        self.fc1 = nn.Dense(10000, 5000)
        #self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(5000, 3755)

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    if device_num>1:
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          parameter_broadcast=True, mirror_mean=True)
        init()
        local_data_url = os.path.join(local_data_url, str(device_id))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True, device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=True)
    mox.file.copy_parallel(args_opt.data_url, local_data_url)

    epoch_size = config.epoch_size
    net = resnet18(class_num=config.class_num)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    if args_opt.do_train:
        dataset = create_dataset(dataset_path=os.path.join(local_data_url, 'train'), do_train=True,
                                 repeat_num=epoch_size, batch_size=config.batch_size)
        eva_dataset = create_dataset(dataset_path=os.path.join(local_data_url, 'test'), do_train=False, batch_size=config.batch_size)
        step_size = dataset.get_dataset_size()

        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

        lr = Tensor(get_lr(global_step=0, lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                           warmup_epochs=config.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                           lr_decay_mode='steps'))
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                       config.weight_decay, config.loss_scale)

        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'})
        time_cb = TimeMonitor(data_size=step_size)
        loss_cb = LossMonitor()
        cb = [time_cb, loss_cb]
        if config.save_checkpoint and (device_num==1 or device_id==0):
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_steps,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix="resnet", directory=local_output_url, config=config_ck)

            cb += [ckpt_cb]
        print("step_size:", step_size)
        model.train(epoch_size, dataset, callbacks=cb)
        if config.save_checkpoint and (device_num==1 or device_id==0):
            # input = np.random.uniform(0.0, 1.0, size=[config.batch_size, 3, config.image_height, config.image_width]).astype(np.float32)
            # serialization.export(net, Tensor(input), file_name='/cache/ckpt/resnet.pb', file_format='GEIR')
            mox.file.copy_parallel(local_output_url, args_opt.train_url)
            res = model.eval(eva_dataset)
            print("result:", res)

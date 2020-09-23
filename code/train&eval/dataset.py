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
"""
create train or eval dataset.
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as dt
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from config import config


def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    device_num = int(os.getenv("RANK_SIZE"))
    rank_id = int(os.getenv("DEVICE_ID"))

    data_path_list = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            data_path_list.append(os.path.join(dataset_path, file))

    schema = dt.Schema()
    schema.add_column('image', de_type=mstype.uint8, shape=[112, 112, 1])  # Binary data usually use uint8 here.
    schema.add_column('label', de_type=mstype.int64, shape=[])


    if device_num == 1:
        ds = dt.TFRecordDataset(data_path_list, num_parallel_workers=4, shuffle=True, schema=schema, shard_equal_rows=True)
    else:
        ds = dt.TFRecordDataset(data_path_list, num_parallel_workers=4, shuffle=True,
                               num_shards=device_num, shard_id=rank_id, schema=schema, shard_equal_rows=True)

    resize_height = config.image_height
    resize_width = config.image_width
    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = C.Resize((resize_height, resize_width))
    rescale_op = C.Rescale(rescale, shift)
    change_swap_op = C.HWC2CHW()

    trans = []

    type_cast_op_train = C2.TypeCast(mstype.float32)
    trans += [resize_op, rescale_op, type_cast_op_train, change_swap_op]
    #trans += [resize_op, rescale_op, type_cast_op_train]
    #trans += [resize_op, type_cast_op_train, change_swap_op]
    type_cast_op = C2.TypeCast(mstype.int32)

    ds = ds.map(input_columns="label", operations=type_cast_op)
    ds = ds.map(input_columns="image", operations=trans)

    ds = ds.shuffle(buffer_size=config.buffer_size)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_num)

    return ds



# coding: utf-8

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import os

latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir='checkpoints')

reader = pywrap_tensorflow.NewCheckpointReader(latest_checkpoint)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key, reader.get_tensor(key).shape)
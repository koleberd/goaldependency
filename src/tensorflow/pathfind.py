#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np

FLAGS = None

actions = {
    'w':0,
    'a':1,
    's':2,
    'd':3,
    'space':4,
    'panUp':5,
    'panDown':6,
    'panRight':7,
    'panLeft':8
}

NUM_ACTIONS = len(actions)

INPUT_TENSOR = [10,2]
LAYER_SIZES = [16,16,16,16,NUM_ACTIONS]


def nn(x):

    dropout = tf.placeholder(tf.float32,name='dropout_chance')

    RS_SIZE = INPUT_TENSOR[0]*INPUT_TENSOR[1]
    reshaped = tf.reshape(x,[1,RS_SIZE])

    layer_0 = fc_layer(reshaped,RS_SIZE,LAYER_SIZES[0],'layer_0')
    layer_1 = fc_layer(tf.nn.relu(layer_0),LAYER_SIZES[0],LAYER_SIZES[1],'layer_1')
    with tf.name_scope('dropout_1'):
        layer_1_dropout = tf.nn.dropout(layer_1,dropout)
    layer_2 = fc_layer(tf.nn.relu(layer_1_dropout),LAYER_SIZES[1],LAYER_SIZES[2],'layer_2')
    layer_3 = fc_layer(tf.nn.relu(layer_2),LAYER_SIZES[2],LAYER_SIZES[3],'layer_3')
    with tf.name_scope('dropout_2'):
        layer_3_dropout = tf.nn.dropout(layer_3,dropout)
    layer_4 = fc_layer(tf.nn.relu(layer_3_dropout),LAYER_SIZES[3],LAYER_SIZES[4],'layer_4')

    return layer_4,dropout

def fc_layer(input_tensor,size_in,size_out,layer_name):
    with tf.name_scope(layer_name):
        w = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1))
        # w = w = tf.Variable(tf.random_normal([size_in,size_out],stddev=0.1))
        b = tf.Variable(tf.truncated_normal([size_out]))
        #b = tf.constant(0.1,shape=[size_out])
    return tf.add(tf.matmul(input_tensor,w),b)


def main():

    nn_input = tf.placeholder(tf.float32, [None, INPUT_TENSOR[0],INPUT_TENSOR[1]])    #IMG_SIZE[0]*IMG_SIZE[1]*COLOR_CHANNELS])
    nn_labels = tf.placeholder(tf.float32, [None, NUM_ACTIONS])
    nn_output, dropout = nn(nn_input)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=nn_labels,logits=nn_output)
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(tf.reduce_mean(cross_entropy))
    with tf.name_scope('accuracy'):
        correct_prediction =  tf.cast(tf.equal(tf.argmax(nn_output, 1), tf.argmax(nn_labels, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import random
from PIL import ImageGrab
from PIL import Image

SIZE_TENSOR = [960,540,3]
COLOR_CHANNELS = SIZE_TENSOR[2]

WIDTH = 1920
HEIGHT = 1080

w_adj = int(WIDTH/4)
h_adj = int(HEIGHT/4)

def main():
    MODEL_DIR = 'trainedModels/'
    MODEL_NAME = 'blockDetector6_stone_False'

    models = os.listdir(MODEL_DIR)
    maxN = models[0]
    maxV = 0
    for m in models:
        if 'meta' in m and float(m.split('.')[1].split('_')[0]) > maxV:
            maxV = float(m.split('.')[1].split('_')[0])
            maxN = m

    MODEL_NAME = maxN[:-5]
    print(MODEL_NAME)

    TEST_DIR = 'training/model_test/'

    model_n = MODEL_DIR + MODEL_NAME
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_n + '.meta')
        saver.restore(sess,model_n)
        n_input = tf.get_collection('n_input')[0]
        n_output = tf.get_collection('n_output')[0]
        n_dropout = tf.get_collection('n_dropout')[0]

        filenames = os.listdir(TEST_DIR)
        filenames = [TEST_DIR + f for f in filenames]

        while True:
            frame = ImageGrab.grab()
            img = frame.crop((w_adj,h_adj,WIDTH-w_adj,HEIGHT-h_adj))
            img = np.reshape(np.array(list(img.getdata())),(1,SIZE_TENSOR[0],SIZE_TENSOR[1],SIZE_TENSOR[2]))
            correct = tf.argmax(n_output,1)
            out = correct.eval(feed_dict={n_input:img,n_dropout:1.0})
            print(out)


main()

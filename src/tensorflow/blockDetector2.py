
"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
FLAGS = None

classes_to_label = {
    'none':0,
    'wood':1,
    'stone':2,
    'iron':3,
    'diamond':4,
    'crafting bench':5,
    'furnace':6,
    'coal':7
}


objects = ['wood','stone','crafting bench']
IMG_DIR = 'training/proc_images/'
SIZE_TENSOR = [960,540,3]
RESIZE_FACTOR = 1
KERNEL_RADII = [5,5]
OUTPUT_CHANNELS = [16,16] #connectedness ebtween conv and pooling layers
FULLY_CONNECTED_SIZE = 128 #1024
BATCH_SIZE = 20
REPEAT_EPOCHS = 1 #10 #times to reuse training batch
NUM_EPOCHS = 10 #20 #times to repeat training
SET_SIZE = 1000








CLASSES2 = len(classes_to_label)*2
CLASSES3 = len(objects) * 2 + 1
CLASSES = 2
IMG_SIZE = [int(SIZE_TENSOR[0]/RESIZE_FACTOR),int(SIZE_TENSOR[1]/RESIZE_FACTOR)]
COLOR_CHANNELS = SIZE_TENSOR[2]
FINAL_LAYER_SIZE = int(IMG_SIZE[1] / 4) * int(IMG_SIZE[0] / 4) * OUTPUT_CHANNELS[1]

def parse_labels(filenames):
    res = []
    for x in filenames:
        name = 0
        if 'stone' not in x or 'OOR' in x:
            name = 1
        o_h = [0 for y in range(0,2)]
        o_h[name] = 1
        res.append(o_h)
    return res


def parse_labels3(filenames):
    res = []
    for x in filenames:
        name = None
        try:
            name = objects.index(x.split('_')[0])*2
            if 'OOR' in x:
                name += 1
        except:
            name = len(objects)*2
        o_h = [0 for y in range(0,len(objects)*2+1)]
        o_h[name] = 1
        res.append(o_h)
    return res

def parse_labels2(filenames):
    res = []
    for x in filenames:
        name = classes_to_label[x.split('_')[0]]*2
        if 'OOR' in x:
            name += 1
        o_h = [0 for y in range(0,len(classes_to_label)*2)]
        o_h[name] = 1
        res.append(o_h)
    return res




def deepnn(x):

    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMG_SIZE[0], IMG_SIZE[1], COLOR_CHANNELS])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([KERNEL_RADII[0], KERNEL_RADII[0], COLOR_CHANNELS, OUTPUT_CHANNELS[0]])
        b_conv1 = bias_variable([OUTPUT_CHANNELS[0]])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([KERNEL_RADII[1], KERNEL_RADII[1], OUTPUT_CHANNELS[0], OUTPUT_CHANNELS[1]])
        b_conv2 = bias_variable([OUTPUT_CHANNELS[1]])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([FINAL_LAYER_SIZE, FULLY_CONNECTED_SIZE])
        b_fc1 = bias_variable([FULLY_CONNECTED_SIZE])
        h_pool2_flat = tf.reshape(h_pool2, [-1, FINAL_LAYER_SIZE])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FULLY_CONNECTED_SIZE, CLASSES])
        b_fc2 = bias_variable([CLASSES])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=COLOR_CHANNELS)
    image_resized = tf.image.resize_images(image_decoded, [IMG_SIZE[0], IMG_SIZE[1]])
    return image_resized, label



def main():




    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE[0], IMG_SIZE[1], COLOR_CHANNELS])    #IMG_SIZE[0]*IMG_SIZE[1]*COLOR_CHANNELS])
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, CLASSES])
    y_conv, keep_prob = deepnn(x)


    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    #graph_location = 'models/graphs' #tempfile.mkdtemp()
    #print('Saving graph to: %s' % graph_location)
    #train_writer = tf.summary.FileWriter(graph_location)
    #train_writer.add_graph(tf.get_default_graph())
    prev_accuracy = 0




    all_files = os.listdir(IMG_DIR)
    all_files = all_files[:int(len(all_files)/BATCH_SIZE)*BATCH_SIZE-1]
    #files = all_files[train_set*SET_SIZE:(train_set+1)*SET_SIZE-1]
    files = []
    while len(all_files) > 0:
        ind = random.randrange(0,len(all_files),1)
        files.append(all_files[ind])
        del all_files[ind]

    #print(files)

    filenames = tf.constant([IMG_DIR + f for f in files])
    labels = tf.constant(parse_labels(files))
    dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    #dataset = dataset.shuffle(buffer_size=len(files))
    dataset = dataset.batch(BATCH_SIZE)
    #dataset = dataset.repeat(REPEAT_EPOCHS)
    batchedSet = dataset.make_one_shot_iterator()
    next_element = batchedSet.get_next()


    print('Beginnning session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #with tf.train.MonitoredTrainingSession() as sess:
        #sess.run(tf.global_variables_initializer())
        #print(int(len(files)/BATCH_SIZE))
        for i in range(0,int(len(files)/BATCH_SIZE)):
            batch = sess.run(next_element)
            print('batch: ' + str(i))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i % 5 == 4:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0] ,y_: batch[1], keep_prob: 1.0})
                print('batch %d, training accuracy %g, delta accuracy %g' % ((i+1), train_accuracy, train_accuracy-prev_accuracy))
                prev_accuracy = train_accuracy
    #((i*BATCH_SIZE)/REPEAT_EPOCHS) % len(files) # number of times full set was used

    #print('test accuracy %g' % accuracy.eval(feed_dict={
        #x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        modelsSaved = os.listdir('trainedModels')
        detectorModels = 0
        for model in modelsSaved:
            if('blockDetector' in model):
                detectorModels += 1
        #saver.save(sess,'trainedModels/blockDetector' + str(detectorModels))

'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''
main()

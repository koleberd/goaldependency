
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
import json
import time as ti

IMG_DIR = 'training/proc_images/'
SIZE_TENSOR = [960,540,3]
RESIZE_FACTOR = 1
KERNEL_RADII = [5,5]
OUTPUT_CHANNELS = [16,16] #connectedness between conv and pooling layers
FULLY_CONNECTED_SIZE = 128 #1024
BATCH_SIZE = 20
VALIDATION_SIZE = 1000
SAMPLE_SIZE = 6000

RNGESUS_ENABLED = False

SUPPORTED_BLOCKS = ['none','wood','stone','crafting bench']

CLASSES = 2
IMG_SIZE = [int(SIZE_TENSOR[0]/RESIZE_FACTOR),int(SIZE_TENSOR[1]/RESIZE_FACTOR)]
COLOR_CHANNELS = SIZE_TENSOR[2]
FINAL_LAYER_SIZE = int(IMG_SIZE[1] / 4) * int(IMG_SIZE[0] / 4) * OUTPUT_CHANNELS[1]



def deepnn(input_tensor):

    with tf.name_scope('reshape'):
        input_image = tf.reshape(input_tensor, [-1, IMG_SIZE[0], IMG_SIZE[1], COLOR_CHANNELS])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([KERNEL_RADII[0], KERNEL_RADII[0], COLOR_CHANNELS, OUTPUT_CHANNELS[0]])
        b_conv1 = bias_variable([OUTPUT_CHANNELS[0]])
        h_conv1 = tf.nn.relu(conv2d(input_image, W_conv1) + b_conv1)

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
        dropout_rate = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout_rate)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FULLY_CONNECTED_SIZE, CLASSES])
        b_fc2 = bias_variable([CLASSES])
        output_tensor = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return output_tensor, dropout_rate


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


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string,channels=COLOR_CHANNELS)
    image_resized = tf.image.resize_images(image_decoded, [IMG_SIZE[0], IMG_SIZE[1]])
    return image_resized, label

def printDataStats(names):
    nameCount = {}
    print('--Sample Breakdown--')

    for fl in names:
        typ = fl.split('_')[0]
        if 'OOR' in fl:
            typ += ' (out of range)'
        if typ in nameCount.keys():
            nameCount[typ] += 1
        else:
            nameCount[typ] = 1

    for item in sorted(nameCount):
        print(str(nameCount[item]) + ' ( ' + str(int(100*nameCount[item]/len(names))) + '% )' + ' - ' + item)
    print(str(len(names)) + ' - total')
    print('(' + str(int(len(names)/BATCH_SIZE)) + ' batches)')
    print('--------------------')

def parse_labels(filenames,target,oor):
    res = []
    for x in filenames:
        name = 0
        if target not in x or ('OOR' in x and not oor) or ('OOR' not in x and oor):
            name = 1
        o_h = [0 for y in range(0,2)]
        o_h[name] = 1
        res.append(o_h)
    return res

#returns shuffled file list
def getFileList(target):
    all_files = os.listdir(IMG_DIR)
    f_2 = []
    for f in all_files:
        if target in f or 'none' in f:
            f_2.append(f)
    all_files = f_2
    all_files = all_files[:int(len(all_files)/BATCH_SIZE)*BATCH_SIZE]
    files = []
    while len(all_files) > 0:
        ind = random.randrange(0,len(all_files),1)
        files.append(all_files[ind])
        del all_files[ind]
    return files

def isPositiveExample(name,target,oor):
    return not (target not in name or ('OOR' in name and not oor) or ('OOR' not in name and oor))

def createBalancedFileLabelSet(target,oor):
    all_files = os.listdir(IMG_DIR)
    positive = []
    negative = []
    for f in all_files:
        if isPositiveExample(f,target,oor):
            positive.append(f)
        elif f.split('_')[0] in SUPPORTED_BLOCKS:
            negative.append(f)

    #make the sets equal sizes
    p_len = len(positive)
    n_len = len(negative)
    if p_len < n_len:
        while len(positive) < n_len:
            positive.extend(positive[0:p_len])
        positive = positive[0:n_len]
    else:
        while len(negative) > p_len:
            negative.extend(negative[0:n_len])
        negative = negative[0:p_len]

    shuffled = []
    while len(positive) > 0:
        ind = random.randrange(0,len(positive),1)
        shuffled.append(positive[ind])
        del positive[ind]
    while len(negative) > 0:
        n_ind = random.randrange(0,len(negative),1)
        s_ind = random.randrange(0,len(shuffled),1)
        shuffled.insert(s_ind,negative[n_ind])
        del negative[n_ind]

    labels = [[int(isPositiveExample(f,target,oor)),int(not isPositiveExample(f,target,oor))] for f in shuffled]
    labels = labels[:int(len(labels)/BATCH_SIZE)*BATCH_SIZE]
    files = [(IMG_DIR + f) for f in shuffled]
    files = files[:int(len(files)/BATCH_SIZE)*BATCH_SIZE]

    return files[:SAMPLE_SIZE],labels[:SAMPLE_SIZE]

def main(target,oor=False):
    input_tensor = tf.placeholder(tf.float32, [None, IMG_SIZE[0], IMG_SIZE[1], COLOR_CHANNELS])
    label_tensor = tf.placeholder(tf.float32, [None, CLASSES])
    output_tensor, dropout_rate = deepnn(input_tensor)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_tensor,logits=output_tensor))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    overall_accuracy = tf.cast(tf.equal(tf.argmax(output_tensor, 1), tf.argmax(label_tensor, 1)), tf.float32)
    accuracy = tf.reduce_mean(overall_accuracy)

    prev_accuracy = 0

    with open('training/trainingTime.json') as ttjs:
        training_times = json.load(ttjs)
    if str(BATCH_SIZE) not in training_times.keys():
        training_times[str(BATCH_SIZE)] = []
    total_time = 0
    for time in training_times[str(BATCH_SIZE)]:
        total_time += time
    if len(training_times[str(BATCH_SIZE)]) != 0:
        total_time /= len(training_times[str(BATCH_SIZE)])

    print('Beginnning session for ' + target + '(oor = ' + str(oor) + ')')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        tf.add_to_collection('n_input',input_tensor)
        tf.add_to_collection('n_output',output_tensor)
        tf.add_to_collection('n_dropout',dropout_rate)
        prev_model_accuracy = 0
        training_round = 0
        while(prev_model_accuracy < .95 and training_round < 1):

            files,labels = createBalancedFileLabelSet(target,oor)
            # CREATE TRAINING SET
            trainingSet = tf.contrib.data.Dataset.from_tensor_slices((files, labels))
            trainingSet = trainingSet.map(_parse_function).batch(BATCH_SIZE)
            next_element_training = trainingSet.make_one_shot_iterator().get_next()
            # CREATE VALIDATION SET
            validationSet = tf.contrib.data.Dataset.from_tensor_slices((files[:VALIDATION_SIZE], labels[:VALIDATION_SIZE]))
            validationSet = validationSet.map(_parse_function).batch(BATCH_SIZE)
            next_element_validation = validationSet.make_one_shot_iterator().get_next()

            #TRAIN THE MODEL
            print('Training round ' + str(training_round) + ' - ' + str(int(len(files)/BATCH_SIZE)) + ' batches at ' + str(total_time) + 's per batch for a total of ' + str(int(len(files)/BATCH_SIZE) * total_time) + 's')
            for i in range(0,50):
                print('=',end='')
            print()
            for i in range(0,int(len(files)/BATCH_SIZE)):
                start_time = ti.time()
                batch = sess.run(next_element_training)
                train_step.run(feed_dict={input_tensor: batch[0], label_tensor: batch[1], dropout_rate: 0.5})
                if i % int((len(files)/BATCH_SIZE)/50) == 0:
                    print('|',end='')
                    sys.stdout.flush()
                if i % int(len(files)/(BATCH_SIZE*8)) == int(len(files)/(BATCH_SIZE*8)) - 1 and RNGESUS_ENABLED:
                    if accuracy.eval(feed_dict={input_tensor: batch[0] ,label_tensor: batch[1], dropout_rate: 1.0}) < .7:
                        raise Exception('RNGesus curses this model')
                training_times[str(BATCH_SIZE)].append(ti.time()-start_time)
            print()

            #VALIDATE THE MODEL
            accuracy_sum = 0
            print('Validating')
            for i in range(0,50):
                print('=',end='')
            print()
            for i in range(0,int(VALIDATION_SIZE/BATCH_SIZE)):
                batch = sess.run(next_element_validation)
                accuracy_sum += accuracy.eval(feed_dict={input_tensor: batch[0] ,label_tensor: batch[1], dropout_rate: 1.0})
                if i % int((VALIDATION_SIZE/BATCH_SIZE)/50) == 0:
                    print('|',end='')
                    sys.stdout.flush()
            print()
            accuracy_sum /= int(VALIDATION_SIZE/BATCH_SIZE)
            print('Accuracy: ' + str(accuracy_sum))
            prev_model_accuracy = accuracy_sum
            training_round += 1

        #SAVE THE MODEL
        m_name = 'blockDetector' + '_' + target + '_' + str(oor) + '_' + str(prev_model_accuracy)[0:5] + '_' + str(SAMPLE_SIZE)
        print('Saving model "' + m_name + '"')
        saver.save(sess,'trainedModels/' + m_name)


        with open('training/trainingTime.json','w') as ttjs:
            json.dump(training_times,ttjs,indent=4)



'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''
#main('wood')
#main('wood',True)
#main('crafting bench')
#main('crafting bench',True)
main('stone')
#main('stone',True)

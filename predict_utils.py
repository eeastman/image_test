import os
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import vgg16
import utils
import csv

LAYER_NAMES = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 
    'conv3_1', 'conv3_2', 'conv3_3', 'pool3', 
    'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 
    'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 
    'fc6', 'fc7', 'fc8']

def loadImages(directory, img_paths):
    '''
    Inputs:
        - directory
        - img_paths
    Outputs:
        tuple of length 3 ---> (batch, batch_size, labels)
        - batch
        - batch_size
        - labels
    '''
    if img_paths != '':
        f = open(img_paths,'r')
        img_paths = f.read().split('\n')
    else:
        img_paths = sorted(os.listdir(directory))
    
    images, labels = [], []
    batch_size = 0

    for path in img_paths:
        if path == '':
            continue
        
        batch_size += 1
        img = utils.load_image(directory + path[:-2])
        images.append(img.reshape((1, 224, 224, 3)))
        labels.append(path[-2:])

    batch = np.zeros((batch_size, 224, 224, 3))
    for i, image in enumerate(images):
        batch[i, :, :, :] = image

    labels = np.array(labels)
    return batch, batch_size, labels

def getLayers(batch, batch_size):
    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            layers = {}

            conv1_1 = sess.run(vgg.conv1_1, feed_dict=feed_dict)
            layers["conv1_1"] = conv1_1

            conv1_2 = sess.run(vgg.conv1_2, feed_dict=feed_dict)
            layers["conv1_2"] = conv1_2

            pool1 = sess.run(vgg.pool1, feed_dict=feed_dict)
            layers["pool1"] = pool1

            conv2_1 = sess.run(vgg.conv2_1, feed_dict=feed_dict)
            layers["conv2_1"] = conv2_1

            conv2_2 = sess.run(vgg.conv2_2, feed_dict=feed_dict)
            layers["conv2_2"] = conv2_2

            pool2 = sess.run(vgg.pool2, feed_dict=feed_dict)
            layers["pool2"] = pool2

            conv3_1 = sess.run(vgg.conv3_1, feed_dict=feed_dict)
            layers["conv3_1"] = conv3_1

            conv3_2 = sess.run(vgg.conv3_2, feed_dict=feed_dict)
            layers["conv3_2"] = conv3_2

            conv3_3 = sess.run(vgg.conv3_3, feed_dict=feed_dict)
            layers["conv3_3"] = conv3_3
        
            pool3 = sess.run(vgg.pool3, feed_dict=feed_dict)
            layers["pool3"] = pool3

            conv4_1 = sess.run(vgg.conv4_1, feed_dict=feed_dict)
            layers["conv4_1"] = conv4_1

            conv4_2 = sess.run(vgg.conv4_2, feed_dict=feed_dict)
            layers["conv4_2"] = conv4_2

            conv4_3 = sess.run(vgg.conv4_3, feed_dict=feed_dict)
            layers["conv4_3"] = conv4_3
            
            pool4 = sess.run(vgg.pool4, feed_dict=feed_dict)
            layers["pool4"] = pool4

            conv5_1 = sess.run(vgg.conv5_1, feed_dict=feed_dict)
            layers["conv5_1"] = conv5_1

            conv5_2 = sess.run(vgg.conv5_2, feed_dict=feed_dict)
            layers["conv5_2"] = conv5_2

            conv5_3 = sess.run(vgg.conv5_3, feed_dict=feed_dict)
            layers["conv5_3"] = conv5_3
            
            pool5 = sess.run(vgg.pool5, feed_dict=feed_dict)
            layers["pool5"] = pool5

            fc6 = sess.run(vgg.fc6, feed_dict=feed_dict)
            layers["fc6"] = fc6

            fc7 = sess.run(vgg.fc7, feed_dict=feed_dict)
            layers["fc7"] = fc7

            fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
            layers["fc8"] = fc8

            prob = sess.run(vgg.prob, feed_dict=feed_dict)

            return layers, prob

def saveLayers(layers, path):
    '''
    Inputs:
        - layers
        - path
    '''
    for key, value in layers.items():
        np.save(os.path.join(path + key + '.npy'),value)

def loadLayers(path):
    '''
    Inputs:
        - 
    Outputs:
        - layers
    '''
    layers = {}
    for layer in LAYER_NAMES:
        layers[layer] = np.load(path + layer + '.npy')

    return layers


def write_accuracies(output_file, results):    
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for layer, accuracies in results.items():
            writer.writerow([layer]+accuracies)

if __name__ == '__main__':
    # write()
    pass

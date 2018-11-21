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
from predict_utils import *


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

def pca(batch_size, vgg_layer, component_used=None):
    '''
    Inputs:
        - batch_size
        - vgg_layer
        - component_used
    Outputs:
        - X_pca
    '''
    # For one pooling layer right now
    X = vgg_layer.reshape((batch_size, -1))

    pca = PCA(n_components=200) # n_components = min(200, n_samples)
    pca.fit(X)
    X_pca = pca.fit_transform(X)

    # print("Variance Explained", pca.explained_variance_ratio_)
    return X_pca if not component_used else X_pca[component_used]

def run_svm(layer, labels, cross_val, trials):
    '''
    Inputs:
        - batch_size
        - labels
        - cross_val
        - layer
        - trials
    Output:
        - accuracy: list of accuracy (float) for each trial
    '''
    
    accuracy = []
    print(layer.size, layer.shape)

    for i in range(trials):
        train_layer, test_layer, y_train, y_test = train_test_split(layer, labels, test_size=0.2)

        X_train = train_layer.reshape((train_layer.shape[0], -1))
        X_test = test_layer.reshape((test_layer.shape[0], -1))
        y_train, y_test = np.array(y_train), np.array(y_test)

        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        # scores = cross_val_score(clf, X_imgs, y_labs, cv=cross_val)
        accuracy.append(clf.score(X_test, y_test))

    return accuracy

def main(directory, img_paths, output_file=None, cross_val=5,
    num_trials=1, use_pca=False, component_used=None):
    '''
    Inputs:
        - directory
        - img_paths
        - output_file
        - cross_val
        - num_trials
        - use_pca
        - component_used
    Output:
        - writes to output file ---> layer, accuracy, std dev
    '''

    batch, batch_size, labels = loadImages(directory, img_paths)

    layers, prob = getLayers(batch, batch_size)
    
    results = {}
    for layer_name, layer in layers.items():

        if use_pca:
            layer = pca(batch_size, layer, component_used)

        accuracy = run_svm(layer, labels, cross_val, num_trials)
        results[layer_name] = accuracy

    print(results)

    write_accuracies(output_file, results)

# Either pass or load layers


if __name__ == '__main__':
    # MAKE SURE ALL PATHS ARE CORRECT
    directory = '/om/user/eeastman/interaction_images/'
    img_paths = '/om/user/eeastman/interaction_imgs.txt' # TEXT FILE WITH LIST OF IMAGE NAMES AND LABELS
                                                        # WHERE EACH LINE HAS "name_of_image.ext label"
    vgg_layers_path = '/om/user/eeastman/vgg_layers_path/'
    pca_layers_path = '/om/user/eeastman/pca_layers_path/'
    
    # FIX SO BACK TO ORIG
    output_file = './interactions_output_predict_5trials_predict5_11-21.csv'

    # PARAMETERS
    use_pca = True
    cross_val = 5
    num_trials = 5
    pca_component_used = None

    main(directory, img_paths, output_file, cross_val, num_trials, use_pca, pca_component_used)

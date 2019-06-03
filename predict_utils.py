import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.switch_backend('agg')

import vgg16
import utils
import csv
import datetime
import random

from scipy import stats

LAYER_NAMES = ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 
    'conv3_1', 'conv3_2', 'conv3_3', 'pool3', 
    'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 
    'conv5_1', 'conv5_2', 'conv5_3', 'pool5', 
    'fc6', 'fc7', 'fc8']



LAYER_NAMES = ['block 1', 'block 2', 'block 3', 'block 4']

LAYER_NAMES = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']

def loadImages(directory, img_paths):
    '''
    Given a directory of images and a txt file of image paths, loads the images into
    numpy objects (batch)

    Parameters:
        - directory: str representing the path to the directory where the images are
        - img_paths: str representing the txt file where each line has the file name of each image

    Returns:
        tuple of length 3 ---> (batch, batch_size, labels)
        - batch: numpy array of images 
        - batch_size: int representing the total number of images
        - labels: numpy array of labels
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

def saveLayers(layers, path):
    '''
    Saves the layers as npy files in the specified path using the
    format: path + layer name (from LAYER_NAMES) + .npy

    Parameters:
        - layers: numpy array representing a layer of the network
        - path: str representing the path to save to

    Returns:
        None
    '''
    for key, value in layers.items():
        np.save(os.path.join(path + key + '.npy'),value)

def loadLayers(path):
    '''
    Loads layers from the given path. This assumes layers are saved using the
    format: path + layer name (from LAYER_NAMES) + .npy

    Parameters:
        - path: str representing the path to the directory where the layers are

    Returns:
        - layers: numpy array of NN layers
    '''
    layers = {}
    for layer in LAYER_NAMES:
        layers[layer] = np.load(path + layer + '.npy')

    return layers

def getLayers(batch, batch_size, vgg_layers_path):
    '''
    Returns layers from either the path (if they were saved previously),
    or running the images through VGG16. If the layers were not previously
    saved, then the layers will be saved in the "vgg_layers_path" directory.

    Parameters:
        - batch: numpy array of images
        - batch_size: int representing the total number of images
        - vgg_layers_path: str representing the path to the directory 
            where the layers are (or should be saved)

    Returns:
        - layers: numpy array of NN layers from VGG16
    '''
    try:
        return loadLayers(vgg_layers_path)

    except:
        pass

    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16('./vgg16-save-5.npy')
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

            # prob = sess.run(vgg.prob, feed_dict=feed_dict)

            saveLayers(layers, vgg_layers_path)

            return layers

def write_accuracies(output_file, results):   
    '''
    Writes accuracies to the given csv file. This assumes that results 
    are of the format (layer name, list of accuracies for each trial)

    Parameters:
        - output_file: str representing the file name (csv) to write to
        - results: dict where the key is layer name and value is a list of 
            accuracies for each trial

    Returns:
        None
    ''' 
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for layer, accuracies in results.items():
            writer.writerow(accuracies)
            #writer.writerow([layer]+accuracies)

def make_error_bar_plot(results, title, fig_name):
    '''
    Makes a bar plot of the accuracy for each layer with error bars over the trials.
    Saves the plot to the "fig_name"

    Parameters:
        - results: dict where the key is layer name and value is a list of 
            accuracies for each trial
        - title: str representing the title of the error bar plot
        - fig_name: str representing the file name (png) to save to

    Returns:
        None
    '''
    means, error = [], []
    x_pos = np.arange(len(LAYER_NAMES))
    y_pos = np.array([0.5 for i in range(len(LAYER_NAMES))])
    # for key in LAYER_NAMES:
    #     layer_list = results[key]
    #     layer_np = np.array(layer_list)
        
    #     means.append(np.mean(layer_np))
    #     error.append(np.std(layer_np))
    #indices = [2,5,9,13,17]
    for i in range(len(LAYER_NAMES)):
    #for i in indices:
        layer_list = results[i]
        print(layer_list.size)
        layer_np = np.array(layer_list)
        
        means.append(np.mean(layer_np))
        error.append(np.std(layer_np))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.plot(x_pos, y_pos, color='red', linestyle='dashed', linewidth=2)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    plt.xticks(rotation=70)
    ax.set_xticklabels(LAYER_NAMES)
    ax.set_title(title)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(fig_name)

def make_error_bar_plot_two(results, results2, title, fig_name):
    '''
    Makes a bar plot of the accuracy for each layer with error bars over the trials.
    Saves the plot to the "fig_name"

    Parameters:
        - results: dict where the key is layer name and value is a list of 
            accuracies for each trial
        - title: str representing the title of the error bar plot
        - fig_name: str representing the file name (png) to save to

    Returns:
        None
    '''
    means, error = [], []
    x_pos = np.arange(len(LAYER_NAMES))
    y_pos = np.array([0.5 for i in range(len(LAYER_NAMES))])
    # for key in LAYER_NAMES:
    #     layer_list = results[key]
    #     layer_np = np.array(layer_list)
        
    #     means.append(np.mean(layer_np))
    #     error.append(np.std(layer_np))
    indices = [2,5,9,13,17]
    #for i in range(len(LAYER_NAMES)):
    for i in indices:
        layer_list = results[i]
        layer_np = np.array(layer_list)
        
        means.append(np.mean(layer_np))
        error.append(np.std(layer_np))

    means2, error2 = [],[]

    #for i in range(len(LAYER_NAMES)):
    for i in indices:
        layer_list = results2[i]
        layer_np = np.array(layer_list)
        
        means2.append(np.mean(layer_np))
        error2.append(np.std(layer_np))

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.bar(x_pos, means2, yerr=error2, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.plot(x_pos, y_pos, color='red', linestyle='dashed', linewidth=2)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    plt.xticks(rotation=70)
    ax.set_xticklabels(LAYER_NAMES)
    ax.set_title(title)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(fig_name)

def t_test_layer(results):
    chance_np = np.array([.5]*20)


    result = []

    indices = [2,5,9,13,17]
    #for i in range(len(LAYER_NAMES)):
    for i in indices:
        layer_list = results[i]
        layer_np = np.array(layer_list)

        t, p = stats.ttest_ind(layer_np, chance_np)

        p /= 2.0

        result.append(p)

    for i in range(len(LAYER_NAMES)):
        print(LAYER_NAMES[i], result[i])


    return result





def build_names(num_trials, use_pca, component_used, leave_out, n=2):
    '''
    TODO: Description

    Parameters:
        - num_trials: int representing the number of trials to run on layer
        - use_pca: bool representing whether or not to use PCA
        - component_used: int representing the component to use for PCA
        - leave_out: bool representing whether or not to use leave_n_out
        - n:

    Returns:
        tuple of length 3 ---> (output_file, plot_title, fig_name)
        - output_file: str representing the file name (csv) to write to
        - plot_title: str representing the title of the error bar plot
        - fig_name: str representing the file name (png) to save to
    '''
    d = datetime.date.today().strftime("%m-%d")

    leave_str = "(leave %s out)" % (str(n)) if leave_out else ''

    if use_pca:
        output_file = './%s_PCA_%strials.csv' % (d, str(num_trials))
        fig_name = '%s_PCA_%strials.png' % (d, str(num_trials))
        plot_title = 'Accuracy for each layer without PCA (top %s components) over %s trials %s' % (str(component_used), str(num_trials), leave_str)
        #plot_title = 'Accuracy for each layer without PCA (50th component) over %s trials' % (str(num_trials))
    else:
        output_file = './%s_without_%strials.csv' % (d, str(num_trials))
        fig_name = '%s_SVM_%strials.png' % (d, str(num_trials))
        plot_title = 'Accuracy for each layer without PCA over %s trials %s' % (str(num_trials), leave_str)
    

    return output_file, plot_title, fig_name

def leave_n_out(layer, labels, n=2, num_scene=4):
    '''
    TODO: Description

    Assumes:
        - 13 different scenes
        - 4 of the same scene (mutual gaze, joint attention, no interact 1, no interact 2)

    Parameters:
        - layer: numpy array of NN layer
        - labels: numpy array of labels associated with each image (index)
        - n: int representing the number of scenes to hold out for test set
        - num_scene: int representing number of images in one scene

    Returns:
        - train_layer: numpy array of NN layer used for training
        - test_layer: numpy array of NN layer used for testing
        - y_train: numpy array of labels associated with each image (index) used for training
        - y_test: numpy array of labels associated with each image (index) used for testing
    '''


    test_indices = [i*2 for i in sorted( random.sample(range(0, 12), n) )]

    all_test_indices = []
    sub = []
    # 13 different scenes
    for i in test_indices:
        all_test_indices.append(i)
        all_test_indices.append(i+1)
        all_test_indices.append(i+(2*13))
        all_test_indices.append(i+1+(2*13))
        sub.append(i)
        sub.append(i+(2*13))

    all_test_indices = sorted(all_test_indices)
    sub = sorted(sub)

    y_test = np.array([l for i, l in enumerate(labels) if i in all_test_indices])
    y_train = np.array([l for i, l in enumerate(labels) if i not in all_test_indices])

    try:
        test_layer = layer[sub[0]:sub[0]+2,...,:]

        for i in sub[1:]:
            test_layer = np.concatenate((test_layer, layer[i:i+2,...,:]), axis=0)

        train_layer = layer[:sub[0], ..., :]

        for i in range(1, len(sub)):
            train_layer = np.concatenate((train_layer, layer[sub[i - 1] + 2:sub[i], ..., :]),
                                         axis=0)

        train_layer = np.concatenate((train_layer, layer[sub[-1] + 2:, ..., :]), axis=0)

    except IndexError:
        test_layer = layer[sub[0]:sub[0] + 2]

        for i in sub[1:]:
            test_layer = np.concatenate((test_layer, layer[i:i + 2]), axis=0)

        train_layer = layer[:sub[0]]

        for i in range(1, len(sub)):
            train_layer = np.concatenate((train_layer, layer[sub[i - 1] + 2:sub[i]]),
                                         axis=0)

        train_layer = np.concatenate((train_layer, layer[sub[-1] + 2:]), axis=0)



    return train_layer, test_layer, y_train, y_test

if __name__ == '__main__':
    # write()
    pass

import os
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score

import vgg16
import utils
import csv


def getPoolingLayers(batch, batch_size):
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

def pca(batch_size, pool_layer, component_used=None):
    '''
    Inputs:
        - batch_size
        - pool_layer
        - component_used
    Outputs:
        - X_pca
    '''
    # For one pooling layer right now
    X = pool_layer.reshape((batch_size, -1))

    pca = PCA(n_components=200) # n_components = min(200, n_samples)
    pca.fit(X)
    X_pca = pca.fit_transform(X)

    # print("Variance Explained", pca.explained_variance_ratio_)
    return X_pca if not component_used else X_pca[component_used]

def run_svm(batch_size, labels, cross_val, layer, trials):
    '''
    Inputs:
        - batch_size
        - labels
        - cross_val
        - layer
        - trials
    Output:
        - total_mean: list of accuracy (float) for each trial
        - total_std: list of std dev (float) for each trial
    '''
    X_imgs = layer
    X_imgs = X_imgs.reshape((batch_size, -1))
    y_labs = np.array(labels)
    total_mean, total_std = [], []
    for i in range(trials):
        clf = svm.SVC(kernel='linear', C=1).fit(X_imgs, y_labs)
        scores = cross_val_score(clf, X_imgs, y_labs, cv=cross_val)
        # print(i, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        total_mean.append(scores.mean())
        total_std.append(scores.std())
    print("AVG Accuracy: %0.2f (+/- %0.2f)" % (sum(total_mean)/trials, (sum(total_std)/trials) * 2)) # TODO

    return (total_mean, total_std)

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
    images = []
    labels = []
    batch_size = 0

    for path in img_paths:
        if path == '':
            continue
        batch_size += 1
        img_path = directory + path[:-2]
        img = utils.load_image(img_path)
        resize = img.reshape((1, 224, 224, 3))
        images.append(resize)
        labels.append(path[-2:])

    batch = np.zeros((batch_size, 224, 224, 3))
    for i, image in enumerate(images):
        batch[i, :, :, :] = image

    return batch, batch_size, labels

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
    
    layers, prob = getPoolingLayers(batch, batch_size)
    
    results = {}
    for layer_name, layer in layers.items():
        final_layer = layer
        if use_pca:
            final_layer = pca(batch_size, laye, component_used)
        mean_acc, std = run_svm(batch_size, labels, cross_val, final_layer, num_trials)
        results[layer_name] = (mean_acc, std)

    print(results)
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            for i in range(len(value[0])):
                writer.writerow([key, value[0][i], value[1][i]])

if __name__ == '__main__':
    # MAKE SURE ALL PATHS ARE CORRECT
    directory = './interaction_images/'
    img_paths = './interaction_imgs.txt' # TEXT FILE WITH LIST OF IMAGE NAMES AND LABELS
                                         # WHERE EACH LINE HAS "name_of_image.ext label"
    output_file = './interactions_output_with_pca.csv'

    # PARAMETERS
    use_pca = True
    cross_val = 5
    num_trials = 20
    component_used = 49

    main(directory, img_paths, output_file, cross_val, num_trials, use_pca, component_used)


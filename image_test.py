import os
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score

import vgg16
import utils


def getPoolingLayers(batch, batch_size):
    # DIRECTORY where all pooling layers will go
    layerPath = './pool_layers/' # CHANGE SO YOU DON'T OVERRIDE

    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            # pool1 = sess.run(vgg.pool1, feed_dict=feed_dict)
            # np.save(os.path.join(layerPath + 'pool1.npy'),pool1)
            
            # pool2 = sess.run(vgg.pool2, feed_dict=feed_dict)
            # np.save(os.path.join(layerPath + 'pool2.npy'),pool2)

            # pool3 = sess.run(vgg.pool3, feed_dict=feed_dict)
            # np.save(os.path.join(layerPath + 'pool3.npy'),pool3)

            # pool4 = sess.run(vgg.pool4, feed_dict=feed_dict)
            # np.save(os.path.join(layerPath + 'pool4.npy'),pool4)

            pool5 = sess.run(vgg.pool5, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'pool5.npy'),pool5)

            return pool5

def pca(batch_size):
    # For one pooling layer right now (easily will update after)
    X = np.load('./pool_layers/pool5.npy')
    X = X.reshape((batch_size, -1))

    pca = PCA(n_components=200)
    pca.fit(X)
    X_pca = pca.fit_transform(X)

    # Make sure there is a "pca_layers" directory or change path
    np.save('./pca_layers/pool5.npy', X_pca)

    return X_pca

def run_svm():
    X_imgs = np.load('./pca_layers/pool5.npy') # According to which pooling layer
    y_labs = np.array([1, 0]) # NEED LABELS 
    # y_labs = np.load('') # INSERT LABEL FILE HERE
    clf = svm.SVC(kernel='linear', C=1).fit(X_imgs, y_labs)
    scores = cross_val_score(clf, X_imgs, y_labs, cv=1) # CHANGE 'CV' WHEN MORE DATA
    # Will print: TODO SAVE?
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def loadImages(directory):
    # Loads images and reshapes to fit for code
    img_paths = os.listdir(directory)
    images = []
    batch_size = 0

    for path in img_paths:
        batch_size += 1
        img = utils.load_image(directory + path)
        resize = img.reshape((1, 224, 224, 3))
        images.append(resize)

    batch = np.zeros((batch_size, 224, 224, 3))
    for i, image in enumerate(images):
        batch[i, :, :, :] = image

    return batch, batch_size

def main():
    # Directory is path of all the images to classify
    directory = './test_data/'
    batch, batch_size = loadImages(directory)
    x = getPoolingLayers(batch, batch_size)
    X_pca = pca(batch_size)
    run_svm()

main()


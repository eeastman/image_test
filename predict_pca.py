import os
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score

import vgg16
import utils


def getPoolingLayers(batch, batch_size):
    layerPath = './pool_layers/' # CHANGE SO YOU DON'T OVERRIDE

    # with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [batch_size, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            pool1 = sess.run(vgg.pool1, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'pool1.npy'),pool1)
            
            pool2 = sess.run(vgg.pool2, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'pool2.npy'),pool2)

            pool3 = sess.run(vgg.pool3, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'pool3.npy'),pool3)

            pool4 = sess.run(vgg.pool4, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'pool4.npy'),pool4)

            pool5 = sess.run(vgg.pool5, feed_dict=feed_dict)
            np.save(os.path.join(layerPath + 'pool5.npy'),pool5)

            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            print(prob)

            return pool5

def pca(batch_size, layer):
    # For one pooling layer right now
    X = np.load('./pool_layers/' + layer)
    X = X.reshape((batch_size, -1))

    pca = PCA(n_components=200) # n_components = min(200, n_samples)
    pca.fit(X)
    X_pca = pca.fit_transform(X)

    np.save('./pca_layers/' + layer, X_pca)

    # variance explained - 90 ... print pca.explained_variance_ratio
    print("Variance Explained", pca.explained_variance_ratio_)
    return X_pca

def run_svm(batch_size, labels, layer):
    X_imgs = np.load('./pca_layers/' + layer)
    X_imgs = X_imgs.reshape((batch_size, -1))
    y_labs = np.array(labels)
    clf = svm.SVC(kernel='linear', C=1).fit(X_imgs, y_labs)
    scores = cross_val_score(clf, X_imgs, y_labs, cv=2) # CHANGE WHEN MORE DATA
    # TODO SAVE
    print(layer)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def loadImages(directory, img_paths):
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
    print('labels', labels)

    batch = np.zeros((batch_size, 224, 224, 3))
    for i, image in enumerate(images):
        batch[i, :, :, :] = image

    return batch, batch_size, labels

def main():
    directory = '/om/data/public/larend/ILSVRC2012_img_val/'
    img_paths = './inds_labels_beagle_tabby.txt'

    # /om/data/public/larend/ILSVRC2012_img_val/ILSVRC2012_val_00033837.JPEG

    batch, batch_size, labels = loadImages(directory, img_paths)

    x = getPoolingLayers(batch, batch_size)

    layers = ['pool1.npy','pool2.npy','pool3.npy','pool4.npy','pool5.npy']
    for layer in layers:
        X_pca = pca(batch_size, layer)
        run_svm(batch_size, labels, layer)

main()



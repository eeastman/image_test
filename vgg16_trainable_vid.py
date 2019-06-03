
import os, datetime
import tensorflow as tf

import vgg16_trainable as vgg16
import utils

from DataLoader import *

# Dataset Parameters
batch_size = 64
load_size = 224
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
#data_mean = np.asarray([103.939, 116.779, 123.68])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 1000
step_display = 10
step_save = 50
path_save = ''
start_from = ''
path_start_np = ''
path_save_np = ''

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '/om/user/eeastman/Moments_in_Time_Raw/training/',   # MODIFY PATH ACCORDINGLY
    'data_list': '/om/user/eeastman/Moments_in_Time_Raw/trainingSet.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '/om/user/eeastman/Moments_in_Time_Raw/validation/',   # MODIFY PATH ACCORDINGLY
    'data_list': '/om/user/eeastman/Moments_in_Time_Raw/validationSet.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int32, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
vgg = vgg16.Vgg16(path_save_np)
vgg.build(x, train_phase)
#logits = vgg16(x, keep_dropout, train_phase)
logits = vgg.fc8

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# loss = tf.reduce_sum((vgg.fc8 - y) ** 2)
# train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    step = 0


    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        print('Step:', step)
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., train_phase: False}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))
        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))

            vgg.save_npy(sess, path_save_np)
        
    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

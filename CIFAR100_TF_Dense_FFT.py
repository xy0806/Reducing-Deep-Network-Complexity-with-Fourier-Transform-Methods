#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:35:38 2017

@author: kiruluta
"""

'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/


        ========>>In the neural network terminology<<===============

One Epoch = one forward pass and one backward pass of all the training examples
Batch Size = the number of training examples in one forward/backward pass.
The higher the batch size, the more memory space you'll need.
Iterations = number of passes, each pass using [batch size] number of examples.
To be clear, one pass = one forward pass + one backward pass (we do not count
the forward pass and backward pass as two different passes).
Example: if you have 1000 training examples, and your batch size is 500, then
it will take 2 iterations to complete 1 epoch.

5/16/2017: before modification to include ffts, accuracy: 94.6% epoch=15, cost=0.84

'''

#from __future__ import print_function

# import CIFAR10 data
from keras.datasets import cifar10
from keras.datasets import cifar100

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import sklearn as sk
from sklearn.preprocessing import normalize
import itertools

'''
def plot_confusion_matrix(array):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
       
    df_cm = pd.DataFrame(array, range(n_classes),
                  range(n_classes))
    #plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 8}) # font size
'''

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    cm=np.around(cm, decimals=2)  # round to 2 decimal precision after decimal
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # apply DropOut to hidden layer
    drop_out = tf.nn.dropout(layer_1, keep_prob)  # DROP-OUT here
    # Hidden layer with RELU activation
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(drop_out, weights['out']) + biases['out']
    return out_layer

# load data
# CIFAR-10
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CIFAR-100
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
n_classes = y_test.shape[1]

# Parameters
learning_rate = 0.001
training_epochs = 300    # best results 250
batch_size = 10000
display_step = 1

# Network Parameters
n_input = 6144  # from flatten 64*32*3 images = 6144 vector for each image
n_hidden_1=int(n_input/2)
n_hidden_2 = 256 # 2nd layer number of features

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.8 at training time
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# saver model instance
saver = tf.train.Saver()

loss=[]       # initialize plotting variables for loss & accuracy
batch_acc=[]
val_acc_list=[]
val_loss_list=[]
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    #--- convert test set to FFT space
    test_fft=np.fft.fftn(X_test, axes=(1,2))           # test data
    test_fft=np.fft.fftshift(test_fft, axes=(1,2))
    test_fft=np.concatenate((test_fft.real,test_fft.imag),axis=1)
    
    # flatten 64*32*3 images to a 6144 vector for each image
    num_pixels = test_fft.shape[1] * test_fft.shape[2] * test_fft.shape[3]
    test_x = test_fft.reshape(test_fft.shape[0], num_pixels).astype('float32')

    # normalize inputs from 0-255 to 0-1
    test_x = test_x / max(test_x.flatten())
    
    #--- convert train set to FFT space
    train_fft=np.fft.fftn(X_train, axes=(1,2))           # test data
    train_fft=np.fft.fftshift(train_fft, axes=(1,2))
    train_fft=np.concatenate((train_fft.real,train_fft.imag),axis=1)
    
    # flatten 64*32*3 images to a 6144 vector for each image
    num_pixels = train_fft.shape[1] * train_fft.shape[2] * train_fft.shape[3]
    train_x = train_fft.reshape(train_fft.shape[0], num_pixels).astype('float32')

    # normalize inputs from 0-255 to 0-1
    train_x = train_x / max(train_x.flatten())
    
    # now split into training batches and labels
    k = [a,b,c,d,e] = np.array_split(train_x, 5, axis=0)
    m = [a,b,c,d,e] = np.array_split(y_train, 5, axis=0)
    #----
    num_of_batches = int(X_train.shape[0]/batch_size)
    # here you can reduce the number of training sample. Defaul total_batch=53
    num_of_batches = 1   # = 5 for entire training sample
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_acc = 0.
        
    
        # Loop over all batches
        for i in range(num_of_batches):
            batch_x, batch_y = k[i], m[i]  # NEED TO MODIFY
            
            # Run optimization op (backprop) and cost op (to get batch loss value)
            # SET keep_prob = 0.2 FOR DROPOUT DURING THIS TIME
            #keep_prob = 0.2
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y, keep_prob: 1.0})
            acc = sess.run(accuracy, feed_dict={x: batch_x,
                                                   y: batch_y, keep_prob: 1.0})
            avg_cost += c
            avg_acc  += acc 
            
        # Compute average loss per epoch
        avg_cost = avg_cost/num_of_batches
        # Compute average accuracy per epoch
        avg_acc = avg_acc/num_of_batches

        loss.append(avg_cost)   # loss per epoch
        batch_acc.append(avg_acc)     # accuracy per epoch

        # This is validation accuracy of the model for each epoch
        #----> FEED 1.0 TO keep_prob DURING VALIDATION/TESTIGN
        #keep_prob = 1.0
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            
        val_acc=accuracy.eval({x: test_x, y: y_test, keep_prob: 1.0})
        val_acc_list.append(val_acc)
        
        _, test_loss = sess.run([optimizer, cost], feed_dict={x: test_x, y: y_test, keep_prob: 1.0})
        val_loss_list.append(test_loss)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "training cost=", \
                "{:.9f}".format(avg_cost), "training acc=",\
                 "{:.5f}".format(avg_acc),"test cost=","{:.5f}".format(test_loss),  "val acc=", "{:.5f}".format(val_acc))
    print("Optimization Finished!")
    #print("batch_x:", batch_x.shape)
    
    # save trained model for later predictions 
    # source: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    checkpoint_file ='saved_model/model.ckpt'
    print("Saving model variables to '%s'." % checkpoint_file)
    saver.save(sess,checkpoint_file)
    
    '''
    #To restore model in prediction use this
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)
        # access a variable from the saved Graph, and so on:
        y_p = tf.argmax(pred, 1)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:y_test, keep_prob: 1.0})
    '''
    
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    y_p = tf.argmax(pred, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:y_test, keep_prob: 1.0})
    #print("Validation Accuracy:", accuracy.eval({x: test_x, y: y_test, keep_prob: 1.0}))
   
    print("validation accuracy:", val_accuracy)
    y_true = np.argmax(y_test,1)   # convert back from hot encoding to [0,1,2,...,9] labels
    
    print("Precision", sk.metrics.precision_score(y_true, y_pred, average='micro'))
    print("Recall", sk.metrics.recall_score(y_true, y_pred, average='micro'))
    print("f1_score", sk.metrics.f1_score(y_true, y_pred, average='micro'))
    print("confusion_matrix")
    cnf_matrix = sk.metrics.confusion_matrix(y_true, y_pred)
    #print('confusion matrix:',cnf_matrix)
    #plot_confusion_matrix(cnf_matrix)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['0','1', '2', '3', '4', '5', '6', '7', '8', '9'],
                      title='Confusion matrix')
    
    # generate class ROC curves
    fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred,pos_label=2)  # label to generate ROC data for
    roc_auc = sk.metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    
    # summarize history for accuracy
    plt.plot(batch_acc)
    plt.plot(val_acc_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train' , 'test' ], loc= 'upper left' )
    plt.show()
    # summarize history for loss
    plt.plot(loss)
    plt.plot(val_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train' ,  'test' ], loc= 'upper left' )
    plt.show()

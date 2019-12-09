import tensorflow as tf
import pandas as pd
import os
import numpy as np

'''
Autoencoder에 필요한 파라메터입니다.
'''

################
# model saver  #
################

SAVER_DIR = "bs752_model"
SAVER_MODEL_NAME = "autoencoder"

################
# layer params #
################
n_inputs = 7
n_hidden1 = 5  # encoder
n_hidden2 = 2  # coding units
n_hidden3 = n_hidden1  # decoder
n_outputs = n_inputs  # reconstruction


################
# train params #
################
learning_rate = 0.01
l2_reg = 0.0005
n_epochs = 20
batch_size = 150
# n_batches = len(train_x) // batch_size
n_batches = 4000

# set the layers using partial
activation = tf.nn.elu
weight_initializer = tf.keras.initializers.he_normal()  # He 초기화
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg)  # L2 규제

inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])

W1_init = weight_initializer([n_inputs, n_hidden1])
W2_init = weight_initializer([n_hidden1, n_hidden2])

# Encoder weights
W1 = tf.Variable(W1_init, dtype=tf.float32, name='W1')
W2 = tf.Variable(W2_init, dtype=tf.float32, name='W2')
# Decoder weights
W3 = tf.transpose(W2, name='W3')  # 가중치 묶기
W4 = tf.transpose(W1, name='W4')  # 가중치 묶기

# bias
b1 = tf.Variable(tf.zeros(n_hidden1), name='b1')
b2 = tf.Variable(tf.zeros(n_hidden2), name='b2')
b3 = tf.Variable(tf.zeros(n_hidden3), name='b3')
b4 = tf.Variable(tf.zeros(n_outputs), name='b4')

hidden1 = activation(tf.matmul(inputs, W1) + b1)
hidden2 = activation(tf.matmul(hidden1, W2) + b2)
hidden3 = activation(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4

# loss
reconstruction_loss = tf.reduce_mean(tf.square(outputs - inputs))
# reg_loss = l2_regularizer(W1) + l2_regularizer(W2)
reg_loss = l2_regularizer(W1)
loss = reconstruction_loss + reg_loss

# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# saver
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, SAVER_MODEL_NAME)
# ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
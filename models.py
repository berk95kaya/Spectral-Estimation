import os
import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
import h5py
from random import randint
import random


##################################################
# SPECTRAL RECONSTRUCTION NETWORK

def prelu( _x, i, reuse = None, custom_init = False, force=None):

    if force != None:
        shap = force
    else:
        shap=_x.get_shape()[-1]
    if custom_init:
        if reuse != None:
            alphas = tf.get_variable('alpha{}'.format(i), shap, initializer=tf.constant_initializer(reuse), dtype=tf.float32)
    else:
        if reuse != None:
            alphas = reuse
        else:
            alphas = tf.get_variable('alpha{}'.format(i),shap, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def residual_hyper_inference(input_tensor_batch, input_channel,  FLAGS):

    deconv_width = 7
    deconv_one = int(deconv_width/2)
    rgb = input_channel
   
#   FSRCNN notation. You can take a look at FSRCNN paper for a deeper explanation
    s, d, m = (128,32,4)

    expand_weight, deconv_weight, upsample_weight = 'w{}'.format(m + 3), 'w{}'.format(m + 4),'w{}'.format(m + 5) 
    weights = {
      'w1': tf.get_variable('w1', shape=[5,5,rgb,s], initializer=tf.contrib.layers.xavier_initializer()),
      'w2': tf.get_variable('w2', shape=[1,1,s,d], initializer=tf.contrib.layers.xavier_initializer()),
      expand_weight: tf.get_variable(expand_weight, shape=[1,1,d,s], initializer=tf.contrib.layers.xavier_initializer()),
      deconv_weight: tf.get_variable(deconv_weight, shape=[5,5,s,FLAGS.c_dim], initializer=tf.contrib.layers.xavier_initializer()),
      upsample_weight: tf.get_variable(upsample_weight,shape=[deconv_width, deconv_width, rgb, FLAGS.c_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
      
    }

    expand_bias, deconv_bias, upsample_bias = 'b{}'.format(m + 3), 'b{}'.format(m + 4), 'b{}'.format(m + 5)
    biases = {
      'b1': tf.Variable(tf.zeros([s]), name='b1'),
      'b2': tf.Variable(tf.zeros([d]), name='b2'),
      expand_bias: tf.Variable(tf.zeros([s]), name=expand_bias),
      deconv_bias: tf.Variable(tf.zeros([FLAGS.c_dim]), name=deconv_bias),
      upsample_bias: tf.Variable(tf.zeros([FLAGS.c_dim]), name=upsample_bias)
    }


    for i in range(3, m + 3):
      weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
      weights[weight_name] = tf.get_variable(weight_name, shape=[3,3,d,d], initializer=tf.contrib.layers.xavier_initializer())
      biases[bias_name] = tf.Variable(tf.zeros([d]), name=bias_name)
      
    conv_feature = prelu(tf.nn.conv2d(input_tensor_batch, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'], 1)

    # Shrinking
    conv_shrink = prelu(tf.nn.conv2d(conv_feature, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'], 2)

    prev_layer = conv_shrink 
    for k in range(3, m + 3):
      weight, bias = weights['w{}'.format(k)], biases['b{}'.format(k)]
      if k == 3:
          prev_layer = prelu(tf.nn.conv2d(prev_layer, weight, strides=[1,1,1,1], padding='VALID') + bias, k)
      if k == 4:
          prev_layer1 = prelu(tf.nn.conv2d(prev_layer, weight, strides=[1,1,1,1], padding='VALID') + bias+tf.slice(conv_shrink, [0,2,2, 0], [-1,tf.shape(conv_shrink)[1]-4,tf.shape(conv_shrink)[2]-4 ,-1]), k)
      if k==5:
          prev_layer = prelu(tf.nn.conv2d(prev_layer1, weight, strides=[1,1,1,1], padding='VALID') + bias, k)
      if k == 6:
          prev_layer = prelu(tf.nn.conv2d(prev_layer, weight, strides=[1,1,1,1], padding='VALID') + bias + 
                                          tf.slice(conv_shrink, [0,4,4,0],[-1,tf.shape(conv_shrink)[1]-8,tf.shape(conv_shrink)[2]-8,-1]) +
                                          tf.slice(prev_layer1, [0,2,2, 0], [-1,tf.shape(prev_layer1)[1]-4,tf.shape(prev_layer1)[2]-4 ,-1]), k)
    
    
    
    # Expanding
    expand_weights, expand_biases = weights['w{}'.format(m + 3)], biases['b{}'.format(m + 3)]
    conv_expand = prelu(tf.nn.conv2d(prev_layer, expand_weights, strides=[1,1,1,1], padding='VALID') + expand_biases + tf.slice(conv_feature, [0,4,4, 0], [-1,tf.shape(conv_feature)[1]-8,tf.shape(conv_feature)[2]-8 ,-1]), 7)
    
    primitive_upsampled = tf.nn.conv2d(input_tensor_batch, weights['w9'], strides=[1,1,1,1], padding='VALID') + biases['b9']

    net =  prelu(tf.nn.conv2d(conv_expand, weights['w8'], strides=[1,1,1,1], padding='VALID') + biases['b8']+ tf.slice(primitive_upsampled, [0,8-deconv_one,8-deconv_one, 0], [-1, tf.shape(input_tensor_batch)[1]-16,tf.shape(input_tensor_batch)[2]-16 ,-1]),8)
    
    return net






###############################################################################
##  ESTIMATOR NETWORK
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv2d_without_activation(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x


def conv2d_valid(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)



def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
    

# Create model
def estimator(x, weights, biases):
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    #x = tf.reshape(x, shape=[-1, size, size, 3])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])       
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)    
    
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3']) 
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)
    
    
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
    conv6 = maxpool2d(conv6, k=2)
    
    conv7 = conv2d(conv6, weights['wc7'], biases['bc7']) 
    conv8 = conv2d(conv7, weights['wc8'], biases['bc8'])
    conv8 = maxpool2d(conv8, k=2)
    
    conv9 = conv2d(conv8, weights['wc9'], biases['bc9'])
    conv10 = conv2d(conv9, weights['wc10'], biases['bc10'])
   
    conv11 = conv2d(conv10, weights['wc11'], biases['bc11'])
    conv12 = conv2d(conv11, weights['wc12'], biases['bc12'])

    return conv12


##############################################################################

def estimator_weights():
       
        # Store layers weight & bias
    var=0.01
    weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, 3, 16],stddev=var)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32],stddev=var)),
        'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64],stddev=var)),
        'wc4': tf.Variable(tf.random_normal([5, 5, 64, 64],stddev=var)),
        
        'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=var)),
        'wc6': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=var)),
        'wc7': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=var)),
        'wc8': tf.Variable(tf.random_normal([3, 3, 64, 64],stddev=var)),
        
        'wc9': tf.Variable(tf.random_normal([3, 3, 64, 128],stddev=var)),
        'wc10': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=var)),
        'wc11': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=var)),
        'wc12': tf.Variable(tf.random_normal([3, 3, 128, 93],stddev=var)),
        
    }
    
    var = 0.001
    biases = {
        'bc1': tf.Variable(tf.random_normal([16],stddev=var)),
        'bc2': tf.Variable(tf.random_normal([32],stddev=var)),
        'bc3': tf.Variable(tf.random_normal([64],stddev=var)),
        'bc4': tf.Variable(tf.random_normal([64],stddev=var)),
        
        'bc5': tf.Variable(tf.random_normal([64],stddev=var)),
        'bc6': tf.Variable(tf.random_normal([64],stddev=var)),
        'bc7': tf.Variable(tf.random_normal([64],stddev=var)),
        'bc8': tf.Variable(tf.random_normal([64],stddev=var)),
        
        'bc9': tf.Variable(tf.random_normal([128],stddev=var)),
        'bc10': tf.Variable(tf.random_normal([128],stddev=var)),
        'bc11': tf.Variable(tf.random_normal([128],stddev=var)),
        'bc12': tf.Variable(tf.random_normal([93],stddev=var)),
    
    
    }

    return weights, biases



# Create model
def classifier(x, weights, biases):
    # Tensor input  4-D: [Batch Size, Height, Width, Channel]
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
   
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)
    
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    return conv4

##############################################################################

def classifier_weights(label_size):
 
    # Store layers weight & bias
    var=0.00001
    weights = {
            'wc1': tf.get_variable('wc1',shape = [5, 5, 3, 256],initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('wc2',shape =[3, 3, 256, 128],initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable('wc3',shape = [3, 3, 128, 64],initializer=tf.contrib.layers.xavier_initializer()),
            'wc4': tf.get_variable('wc4',shape = [3, 3, 64, label_size],initializer=tf.contrib.layers.xavier_initializer()),
        }
        
    biases = {
            'bc1': tf.Variable(tf.random_normal([256],stddev= var)),
            'bc2': tf.Variable(tf.random_normal([128],stddev= var)),
            'bc3': tf.Variable(tf.random_normal([64],stddev= var)),
            'bc4': tf.Variable(tf.random_normal([label_size],stddev= var)),     
        }
    return weights, biases





              



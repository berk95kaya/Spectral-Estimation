import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
from random import randint
import math
from tensorflow.python.framework import ops
ops.reset_default_graph()
import h5py
import random
import glob
import pickle
import sys
from skimage.measure import compare_ssim as ssim
from functions import *
from models import estimator, estimator_weights
#import matplotlib.pyplot as plt


flags = tf.app.flags
flags.DEFINE_string("name", "es1", "Name of the experiment ")
flags.DEFINE_integer("num_steps",500000,"number of iterations")
flags.DEFINE_float("learning_rate",0.00001,"learning rate of the optimizer")
flags.DEFINE_integer("c_dim",31,"number of channels in the spectral data")
flags.DEFINE_integer("alpha",8,"normalization coef for sensitivities")
flags.DEFINE_string("sens_dir", "./", "Folder which contains sensitivity dataset")
flags.DEFINE_string("discrete_set", "sensitivities_8.mat", "Name of the mat file which contains sensitivity functions")
flags.DEFINE_integer("display_step",20000," number of steps to validate results")
flags.DEFINE_float("Li",1,"Image loss coefficient")
flags.DEFINE_float("Ll",0.00001,"Label loss coefficient")
flags.DEFINE_float("Ls",0.001,"Smoothness coefficient")
flags.DEFINE_integer("training_error_step",2000," number of steps to calculate training error")


# OPERATION MODES
flags.DEFINE_string("sens_type",'continuous', "continous or discrete")
flags.DEFINE_boolean("training", True, "True if training is performed. Otherwise, only testing is performed")
flags.DEFINE_string("dataset", "ICVL", "ICVL or CAVE")


FLAGS = flags.FLAGS

  

modelfolder="/media/berk/DATA/Proje/Models/"
HSIfolder="/media/berk/DATA/Proje/Hyperspectral_Dataset_sized/resim_sized "

testfilename = 'Estimator'+FLAGS.name+'.txt'
modelname='Estimator'+FLAGS.name+'.ckpt'

try:
    training_list = np.load('training_number.npy')
    training_length = len(training_list)
    validation_list = np.load('validation_number.npy')
    validation_length = len(validation_list)
    test_list = np.load('test_number.npy')
    test_length = len(test_list)
except:
    print("Create train/validation/test sets before using this code")
    sys.exit()


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

    
X = tf.placeholder(tf.float32, [None,  None, None, 3])
Y = tf.placeholder(tf.float32, [FLAGS.c_dim,3])
H = tf.placeholder(tf.float32, [ None, None, FLAGS.c_dim])

weights, biases = estimator_weights()
estimation = estimator(X, weights, biases)

estimation  = tf.reduce_mean( estimation, 0)
estimation  = tf.reduce_mean( estimation, 0)
estimation  = tf.reduce_mean( estimation, 0)


image_predicted = tf.matmul(tf.reshape(H,[tf.shape(H)[0]*tf.shape(H)[1],FLAGS.c_dim]), tf.transpose(tf.reshape(estimation, [3,FLAGS.c_dim])))
image_real = tf.matmul(tf.reshape(H,[tf.shape(H)[0]*tf.shape(H)[1],FLAGS.c_dim]),Y)

a=np.ones([FLAGS.c_dim])*2
b=np.ones([FLAGS.c_dim-1])*(-1)
A = tridiag(b, a, b)
A[0,:]=0
A[FLAGS.c_dim-1,:]=0
R = tf.convert_to_tensor(A, dtype=tf.float32)
smoothness_penalty=tf.reduce_sum(tf.square(tf.matmul(R,tf.transpose(tf.reshape(estimation,[3,FLAGS.c_dim])))))
label_penalty= tf.reduce_sum(tf.squared_difference(tf.transpose(tf.reshape(estimation, [3,FLAGS.c_dim])), Y))
image_penalty= tf.reduce_mean(tf.squared_difference(image_predicted, image_real)) 
loss_op = FLAGS.Li* image_penalty + FLAGS.Ll* label_penalty +  FLAGS.Ls*smoothness_penalty

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = np.copy(FLAGS.learning_rate)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,50000, 0.93, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate,epsilon=1e-6).minimize(loss_op, global_step = global_step) 


init = tf.global_variables_initializer()
 
minimum_loss = 1e34
saver = tf.train.Saver()


losses = np.zeros((FLAGS.training_error_step))


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Run the initializer
    sess.run(init)
    for step in range(0, FLAGS.num_steps):
       # START THE LOOP
        HSI = read_HSI_sized( HSIfolder +'('+ str(   training_list[step % training_length]) + ')', FLAGS  )
        HSI_expanded = np.expand_dims(HSI, axis = 0)
        label = create_sensitivity(FLAGS) 
        input_image = create_rgb( HSI_expanded , label)
                       
        _, err = sess.run([train_op,image_penalty], feed_dict={X: input_image, Y: label, H:HSI })
        losses[step % FLAGS.training_error_step] = err
            
        # Validation
        if step % (FLAGS.display_step) == 0:
                file = open(testfilename,"a")
                print("\n")
                
                # TRAINING ERROR CALCULATION
                average_loss = sum(losses)/FLAGS.training_error_step
                print("Step " + str(step) + ",Training Loss= " + str(average_loss) )
                writing = ("\n Step " + str(step) + ",Training Loss= " + str(average_loss)  )
                file.write(writing)
                
                ##############################
                total_loss=0
                average_psnr=0
                psnr_value=0
                for k in range(0,validation_length):
               
                    HSI = read_HSI_sized( HSIfolder+'(' + str(validation_list[step % validation_length])+')' , FLAGS )

                    HSI_expanded = np.expand_dims(HSI, axis = 0)
                    data_y = create_sensitivity(FLAGS)  
                    data_x = create_rgb( HSI_expanded , data_y)
                    
                    output= sess.run(estimation, feed_dict={X: data_x})
                    psnr_value, loss, ssim_,  mrae_ = evaluation(output, data_y,  HSI , FLAGS)
                    average_psnr += psnr_value/validation_length
                    
                    total_loss += loss
                average_loss = total_loss / validation_length
                print("Step " + str(step) + ",Validation Loss= " + str(average_loss) )
                print("Step " + str(step) + ",Average PSNR= " + str(average_psnr) )
                writing = ("\n \n Step " + str(step) + ", Validation Loss= " + str(average_loss) )
                file.write(writing)
                writing = ("\n Step " + str(step) + ", Average PSNR= " + str(average_psnr) )
                file.write(writing)
                writing = ("\n Step " + str(step) + ", Best Validation loss = " + str(minimum_loss) )
                file.write(writing)
                file.close()
                
                if average_loss < minimum_loss:
                    saver.save(sess, modelfolder+ modelname)
                    file_name = open('weights_estimator_'+ FLAGS.name + '.obj', 'wb')
                    weights_n = pickle.dump(sess.run(weights),file_name)
                    file_name = open('biases_estimator_'+ FLAGS.name + '.obj', 'wb')
                    biases_n = pickle.dump(sess.run(biases),file_name)
                    minimum_loss = average_loss
    
                   
    print("Optimization Finished!")        
        # Testing loop
        
    mse_results = []
    rmse_results = []
    psnr_results=[]
    ssim_results=[]
    mrae_results = []
    average_mse =[]
    
    file_name = open('weights_estimator_'+ FLAGS.name + '.obj', 'rb')
    weights_n = pickle.load(file_name)
    file_name = open('biases_estimator_'+ FLAGS.name + '.obj', 'rb')
    biases_n = pickle.load(file_name)
    
    k=0
    while  (len(mse_results)<1000):
        print(k)

        HSI = read_HSI_sized(  HSIfolder+'(' + str(test_list[step % test_length])+')' , FLAGS  )

        HSI_expanded = np.expand_dims(HSI, axis = 0)
          
        data_y = create_sensitivity(FLAGS)                         
        data_x = np.float32(create_rgb( HSI_expanded , data_y)) 
        
        output= sess.run(estimator(data_x,weights_n,biases_n))
         
        psnr_value , loss, ssim_, mrae_ = evaluation(output, data_y,  HSI , FLAGS )
        
        mse_results.append(loss)
        psnr_results.append(psnr_value)
        rmse_results.append(math.sqrt(loss))
        ssim_results.append(ssim_)
        mrae_results.append(mrae_)
        average_mse.append(np.mean(mse_results))
        k = k+1




    print("Average MSE :" + str(np.mean(mse_results)))
    print("Average RMSE :" + str(np.mean(rmse_results)))
    print("Average PSNR :" + str(np.mean(psnr_results)))
    print("Average SSIM :" + str(np.mean(ssim_results)))
    print("Average MRAE :" + str(np.mean(mrae_results)))
    print("Testing is done!!")

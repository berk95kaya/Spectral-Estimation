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
from models import classifier, classifier_weights

#import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string("name", "classification_experiment", "Name of the experiment ")
flags.DEFINE_integer("num_steps",100000,"number of iterations")
flags.DEFINE_integer("num_of_sens",3,"number of sensitivity function used in classification (2-40)")
flags.DEFINE_float("learning_rate",0.0001,"learning rate of the optimizer")
flags.DEFINE_integer("c_dim",31,"number of channels in the spectral data")
flags.DEFINE_string("sens_dir", "./", "Folder which contains sensitivity dataset")
flags.DEFINE_string("discrete_set", "sensitivities_8.mat", "Name of the mat file which contains sensitivity functions")
flags.DEFINE_integer("display_step",200," number of steps to validate results (>100)")
flags.DEFINE_integer("training_error_step",200," number of steps to calculate training error (<= display step)")
flags.DEFINE_string("dataset", "ICVL", "ICVL or CAVE")
flags.DEFINE_boolean("training", True , "True if training is performed. Otherwise, only testing is performed")
flags.DEFINE_string("sens_type",'discrete', "it must be discrete")

FLAGS = flags.FLAGS

# This process selects the sensitivities randomly


modelfolder="/media/berk/DATA/Proje/Models/"
HSIfolder="/media/berk/DATA/Proje/Hyperspectral_Dataset_sized/resim_sized "

testfilename = 'class_'+FLAGS.name+'.txt'
modelname='class_'+FLAGS.name+'.ckpt'

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


label_size = FLAGS.num_of_sens


X = tf.placeholder(tf.float32, [None,  None, None, 3])
Y = tf.placeholder(tf.float32, [None, label_size])
H = tf.placeholder(tf.float32, [ None, None, FLAGS.c_dim])

weights, biases = classifier_weights(label_size)
classification = classifier(X, weights, biases)

classification  = tf.reduce_mean( classification, 0)
classification  = tf.reduce_mean( classification, 0)
classification  = tf.reduce_mean( classification, 0)
prediction = tf.nn.softmax(classification)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classification, labels=Y))

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = np.copy(FLAGS.learning_rate)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,1000, 0.93, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate,epsilon=1e-6).minimize(loss_op, global_step = global_step) 


correct_pred = tf.equal(tf.argmax(prediction), tf.argmax(Y, 1))


init = tf.global_variables_initializer()
 
minimum_loss = 1e34
saver = tf.train.Saver()


losses = np.zeros((FLAGS.training_error_step))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    if FLAGS.training:
        sens_list = np.arange(1,41)
        random.shuffle(sens_list)
        # otherwise define sens_list in array format
        np.save("sens_list_"+FLAGS.name, sens_list)

        # Run the initializer
        file = open(testfilename,"a")
        writing = ( str(sens_list[0:FLAGS.num_of_sens])  )
        file.write(writing)
        file.close()
        print(sens_list[0:FLAGS.num_of_sens])
        sess.run(init)
        
        for step in range(1, FLAGS.num_steps):
            # START THE LOOP
            HSI = read_HSI_sized( HSIfolder +'('+ str(   training_list[step % training_length]) + ')', FLAGS  )
            HSI_expanded = np.expand_dims(HSI, axis = 0)
            sens_index = randint(0,FLAGS.num_of_sens-1)
            sens = create_sensitivity(FLAGS , sens_list[sens_index])
            data_y = np.zeros((1,label_size))
            data_y[0,sens_index] = 1  # Hatali
            data_x = create_rgb( HSI_expanded , sens)
            
            _ , err = sess.run([train_op, loss_op], feed_dict={X: data_x, Y: data_y, H:HSI })
            losses[step % FLAGS.training_error_step] = err
    
            ##########################
            # Evaluation part 
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
                accuracy = 0
                for k in range(0,validation_length):
                    
                    HSI = read_HSI_sized( HSIfolder + "(" + str(validation_list[k % validation_length]) + ")", FLAGS)
                    HSI_expanded = np.expand_dims(HSI, axis = 0)
                    sens_index = randint(0,FLAGS.num_of_sens-1)
                    sens = create_sensitivity(FLAGS,sens_list[sens_index]) 
                    data_y = np.zeros((1,label_size))
                    data_y[0,sens_index] = 1
                    data_x = create_rgb( HSI_expanded , sens)
                            
                    pred  = sess.run(prediction, feed_dict={X: data_x})

                    
                    label_predicted = np.argmax(pred)
                    sens_reconstructed = create_sensitivity(FLAGS,  label_predicted)
                    label_real=np.argmax(data_y)
                    sens_reconstructed2 = create_sensitivity(FLAGS, label_real)
                    
                    a = [label_predicted == label_real]
    
                    
                    _, loss, ssim_, mrae_ = evaluation(sens_reconstructed, sens_reconstructed2, HSI, FLAGS)
                    
                                    
                    total_loss += loss
                    if label_predicted == label_real:
                        accuracy = accuracy + 100 / validation_length
    
                average_loss = total_loss / validation_length
                print("Step " + str(step) + ",Validation Loss= " + str(average_loss) )
                print("Step " + str(step) + ",Accuracy= " + str(accuracy) )
                
                writing = ("\n Step " + str(step) + ", Validation Loss= " + str(average_loss) )
                file.write(writing)
                writing = ("\n Step " + str(step) + ",Accuracy= " + str(accuracy) )
                file.write(writing)
                writing = ("\n Step " + str(step) + ", Best Validation loss = " + str(minimum_loss) )
                file.write(writing)
                
                
                if average_loss < minimum_loss:
                    saver.save(sess, modelfolder+ modelname)
                    file_name = open('weights_classifier_'+ FLAGS.name + '.obj', 'wb')
                    weights_n = pickle.dump(sess.run(weights),file_name)
                    file_name = open('biases_classifier_'+ FLAGS.name + '.obj', 'wb')
                    biases_n = pickle.dump(sess.run(biases),file_name)
                    minimum_loss = average_loss
            
                   
        print("Optimization Finished!")

        
        
    # Testing Loop
    accuracy = 0
    mse_results = []
    rmse_results = []
    psnr_results=[]
    ssim_results=[]
    mrae_results = []



    file_name = open('weights_classifier_'+ FLAGS.name + '.obj', 'rb')
    weights_n = pickle.load(file_name)
    file_name = open('biases_classifier_'+ FLAGS.name + '.obj', 'rb')
    biases_n = pickle.load(file_name)

    sens_list = np.load("sens_list_"+FLAGS.name+".npy")

    for k in range(0,test_length):
        for l in range (0, FLAGS.num_of_sens):
        
            HSI = read_HSI_sized( HSIfolder + "(" + str(test_number[k % test_length]) + ")")
            HSI_expanded = np.expand_dims(HSI, axis = 0)
            sens_index = randint(0,FLAGS.num_of_sens-1)
            sens = create_sensitivity(FLAGS, sens_list[sens_index]) 
            data_y = np.zeros((1,label_size))
            data_y[0,sens_index] = 1
            data_x = create_rgb( HSI_expanded , sens)
                    
            pred  = sess.run(prediction, feed_dict={X: data_x})

            
            label_predicted = np.argmax(pred)
            sens_reconstructed = create_sensitivity(FLAGS, label_predicted)
            label_real=np.argmax(data_y)
            sens_reconstructed2 = create_sensitivity(FLAGS, label_real)
            
            psnr_value , loss, ssim_, mrae_ = evaluation(sens_reconstructed, sens_reconstructed2, HSI, FLAGS)
            print(label_predicted)
            print(label_real)
            print("\n")
            a = [label_predicted == label_real]
            mse_results.append(loss)
            psnr_results.append(psnr_value)
            rmse_results.append(math.sqrt(loss))
            ssim_results.append(ssim_)
            mrae_results.append(mrae_)
         
            if label_predicted == label_real:
                accuracy = accuracy + 100 / (FLAGS.num_of_sens*test_length)

    print("Accuracy = " + str(accuracy) )
    print("Average MSE :" + str(np.mean(mse_results)))
    print("Average RMSE :" + str(np.mean(rmse_results)))
    print("Average PSNR :" + str(np.mean(psnr_results)))
    print("Average SSIM :" + str(np.mean(ssim_results)))
    print("Average MRAE :" + str(np.mean(mrae_results)))
    print("Testing is done!!")

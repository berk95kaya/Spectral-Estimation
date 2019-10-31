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
import time
from functions import *
from models import residual_hyper_inference



flags = tf.app.flags
flags.DEFINE_string("name", "generic", "Name of the experiment ")
flags.DEFINE_integer("num_steps",1000000,"number of iterations")
flags.DEFINE_float("learning_rate",0.0005,"learning rate of the optimizer")
flags.DEFINE_integer("c_dim",31,"number of channels in the spectral data")
flags.DEFINE_integer("alpha",8,"normalization coef for sensitivities")
flags.DEFINE_string("sens_dir", "./", "Folder which contains sensitivity dataset")
flags.DEFINE_string("discrete_set", "sensitivities_8.mat", "Name of the mat file which contains sensitivity functions")
flags.DEFINE_integer("display_step",20000," number of steps to validate results")
flags.DEFINE_integer("training_error_step",2000," number of steps to calculate training error")


# OPERATION MODES
flags.DEFINE_string("sens_type",'continuous', "continous or discrete")
flags.DEFINE_boolean("training", True, "True if training is performed. Otherwise, only testing is performed")
flags.DEFINE_string("dataset", "ICVL", "ICVL or CAVE")


FLAGS = flags.FLAGS


pad=8
stride = 32
input_channel= 3

loadCheckPoint = True
load_checkpoint_dir = "best_checkpoint"
train_log_file= "train_generic_log.txt"
vali_log_file= "vali_generic_log.txt"

checkpoint_dir = "/media/berk/DATA/Proje/check/"
best_checkpoint_dir = "/media/berk/DATA/Proje/best_checkpoint/"
output_dir = "/media/berk/DATA/Proje/output_dir/"
train_batches_dir = "/media/berk/DATA/Proje2/batches/"
sens_dir = "./"
HSIfolder="/media/berk/DATA/Proje/Hyperspectral_Dataset/resim "
 

sens_list = np.arange(1,41)
random.shuffle(sens_list)

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


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(best_checkpoint_dir):
    os.makedirs(best_checkpoint_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(train_batches_dir):
    os.makedirs(train_batches_dir)
    
    
    
def save(model_name,  checkpoint_dir, step):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)

def load( checkpoint_dir):
    #print(" [*] Reading checkpoints...")
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        #print("Found")
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print("Loaded")
        #print(str(ckpt) + " and : " + str(ckpt_name))
        return True
    else:
        print("Failed")
        return False
    
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
 
images = tf.placeholder(tf.float32, [None, None, None, input_channel], name='images')
labels = tf.placeholder(tf.float32, [None, None, None, FLAGS.c_dim], name='labels')
pred = residual_hyper_inference(images, input_channel, FLAGS)
loss = tf.reduce_mean(tf.squared_difference(pred , labels))
saver = tf.train.Saver()

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = np.copy(FLAGS.learning_rate)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,50000, 0.93, staircase=True)
train_op = tf.train.AdamOptimizer(learning_rate,epsilon=1e-6).minimize(loss, global_step = global_step)

with tf.Session(config=config) as sess:
    tf.initialize_all_variables().run()

    start_time = time.time()
    print("Beginning batch training setup...")

    if loadCheckPoint:
        load(load_checkpoint_dir)
        
    counter = 0
    losses = np.zeros((1000))
    best_rgb_val_loss = (10000,0)

    print("Training...")
    start_time = time.time()


    batches  = glob.glob(train_batches_dir+'*')
    n_batch = len(batches)
    print("Len of batches: "+str(n_batch))

    batch_average = 0
    
    
    for iteration in range(FLAGS.num_steps):
        
        data_dir = batches[iteration % n_batch]
        train_data = read_batch(data_dir) 

        sens = create_sensitivity(FLAGS) 
        train_rgb = create_rgb( train_data, sens)
        train_data_cropped = train_data[:,pad:train_data.shape[1]-pad,pad:train_data.shape[2]-pad, : ] 

        counter += 1
        _, err = sess.run([train_op,loss], feed_dict={images: train_rgb, labels: train_data_cropped})
            
        batch_average += err
    
                
        if counter % FLAGS.display_step == 0:
            save(FLAGS.name, checkpoint_dir, counter)
        losses[counter % 1000] = err
            
    
        ########### VALIDATION
        if (counter % FLAGS.display_step == 0) :
              GRMSEs=[]
              GrRMSEs=[]
              ARMSEs = []
              ArRMSEs = []
              
              uGRMSEs=[]
              uGrRMSEs=[]
              uARMSEs = []
              uArRMSEs = []
             
              for k in range(validation_length):
                    
                      data_index = validation_list[k]
                      HSI = read_HSI(HSIfolder + "(" + str(data_index)+").mat" , FLAGS)
                      test_data = np.expand_dims(HSI, axis = 0)
                      
                      sens = create_sensitivity(FLAGS) 
                      
                      test_rgb = create_rgb(test_data, sens)
                      test_data = np.squeeze(test_data)
                      test_rgb = np.squeeze(test_rgb)
                      
                      test_data = test_data[pad:test_data.shape[0]-pad,pad:test_data.shape[1]-pad, : ]         
                                              
                      
                      res_parts = []
                      for piece in range(4):
                          part = deconstruct(test_rgb,piece,pad)
                          
                          res_part = pred.eval(feed_dict={images: np.reshape(part, (1,part.shape[0],part.shape[1],input_channel))})
                          res_parts.append(res_part)
                      
                      result = construct(res_parts,(test_rgb.shape[0]-2*pad,test_rgb.shape[1]-2*pad,FLAGS.c_dim))
                      result = result.squeeze()
                      fin_res = result
               
                              
                      ########################     
                      temp = test_data.squeeze()
                      temp_result = (fin_res)
                      temp_temp = (temp)
                    
                      grmse, grrmse, armse,arrmse = psnr_calculation(temp_result,temp_temp , False)
                      GRMSEs.append(grmse)
                      GrRMSEs.append(grrmse)
                      ARMSEs.append(armse)
                      ArRMSEs.append(arrmse)
                      
                      ugrmse, ugrrmse, uarmse,uarrmse = psnr_calculation(temp_result,temp_temp , True)
                      uGRMSEs.append(ugrmse)
                      uGrRMSEs.append(ugrrmse)
                      uARMSEs.append(uarmse)
                      uArRMSEs.append(uarrmse)
                      
                      print("Validated image : " + str(k)+"_sens")
    
        
              temp_float = sum(GRMSEs)/validation_length
              print("Current RMSE: " + str(temp_float))
              print("RMSE best : " + str(best_rgb_val_loss[0]))
              if  temp_float < best_rgb_val_loss[0]:
                      best_rgb_val_loss = (temp_float, counter) 
                     
                      save(FLAGS.name,best_checkpoint_dir, counter)
                      
              text_file = open(vali_log_file, "a")
              text_file.write("\nAverage Galiani RMSE " + " : "+str(counter) + " : "+str(sum(GRMSEs)/validation_length ))
              text_file.write("\nMax RMSE : " + str(best_rgb_val_loss[1]) +" : " + str(best_rgb_val_loss[0]))      
              text_file.close()    
              print("\nAverage PSNR Validation : " + str(counter) +" : " + str(sum(GRMSEs)/validation_length ))
      
        if counter % 1000 == 0:
              text_file = open(train_log_file, "a")
              text_file.write("\nError in step " + str(counter) +" : " + str(sum(losses)/1000))
              text_file.close()
              print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                % ((FLAGS.num_steps+1), counter, time.time() - start_time, sum(losses)/1000))
              

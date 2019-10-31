
import os
import tensorflow as tf
import os
import numpy as np
import scipy.io as sio
import h5py
from random import randint
import random
import math
from skimage.measure import compare_ssim as ssim
from forecasting_metrics import mrae





#################################
# DATA IMPORT

def read_HSI(HSI_file, FLAGS):

    if FLAGS.dataset is 'ICVL':
        f = h5py.File(HSI_file, 'r')
        x = f["rad"]
        rad=np.float32(x)
        rad=np.swapaxes(rad,0,2) / 4095
    elif FLAGS.dataset is 'CAVE':
        rad = np.load(HSI_file)
    else:
        sys.exit('Change the name of the dataset')
    return rad 

 
    
def read_HSI_sized(HSI_file, FLAGS):
    if FLAGS.dataset == "ICVL":
        f = sio.loadmat(HSI_file)
        f_data=f['data']
        f_rad=f_data['rad']
        rad=f_rad[0,0]
    elif FLAGS.dataset == "CAVE":
        rad = np.load(HSI_file)
    else:
        sys.exit('Change the name of the dataset')
    return rad
    

def read_batch(data_dir):
    f = h5py.File(data_dir )
    batch = f["batch"]
    return batch
    

#########################################333333
# SENSITIVITY 



def create_sensitivity(FLAGS, i=None):
    if FLAGS.sens_type is 'discrete':
        f = sio.loadmat(FLAGS.sens_dir + str(FLAGS.discrete_set))
        f=f["sensitivities"]
        sens = f["sens"+str(i)]  
        return sens[0,0]
    elif FLAGS.sens_type is 'continuous':
        sens = create_sens_mixture()
        return sens
    else:
        print('Unknown sensitivity set.  Use either continuous or discrete ')

def create_sens_gaussian():
    x = np.arange(1,32)

    mean_red = np.random.uniform(low = 16 ,high = 26 )
    mean_green = np.random.uniform(low = 10 ,high = 20 )
    mean_blue = np.random.uniform(low = 5 ,high = 15 )
    
    sigma_red = np.random.uniform(low = 2 ,high = 6 )
    sigma_green = np.random.uniform(low = 2 ,high = 6 )
    sigma_blue = np.random.uniform(low = 2 ,high = 6 )
    
    y_red = np.exp(  -(x-mean_red)**2 / sigma_red**2)
    y_green = np.exp(  -(x-mean_green)**2 / sigma_green**2)
    y_blue = np.exp(  -(x-mean_blue)**2 / sigma_blue**2)
    
    sens = np.transpose(np.concatenate(([y_red],  [y_green], [y_blue]),axis=0))/8
    
    return sens

def create_sens_mixture():
    k= randint(1,5)
    r = [random.random() for i in range(0,k)]
    r = r/np.sum(r)
    sens = np.zeros([31,3])
    for i in range(k):
        sens = sens+ r[i]*create_sens_gaussian() 
    return sens


def create_rgb( data, sens):
      n = data.shape[0]
      img = np.zeros([data.shape[1],data.shape[2],data.shape[3]] )
      rgb= np.zeros([n, data.shape[1],data.shape[2], 3 ])

      for k in range(0,n):
          img[:,:,:] = data[k,:,:,:]
          rgb[k,:,:,0:3]=hyper_to_rgb(img,img.shape[0],img.shape[1],img.shape[2], sens)
      return rgb




def create_rgb_with_sens( data, sens):
      n = data.shape[0]
      img = np.zeros([data.shape[1],data.shape[2],data.shape[3]] )
      rgb= np.zeros([n, data.shape[1],data.shape[2], 96 ])
      tmp = np.reshape(np.transpose(sens), [93])

      for k in range(0,n):
          img[:,:,:] = data[k,:,:,:]
          rgb[k,:,:,0:3]=hyper_to_rgb(img,img.shape[0],img.shape[1],img.shape[2], sens)
          rgb[k,:,:,3:(93+3)] = tmp
      return rgb

  
def hyper_to_rgb(rad,h,w,numbands, cie):
    # Converts hyperspectral image into rgb
    rad2d=np.reshape(rad,[h*w,numbands])
    rgb2d=np.dot(rad2d,cie)
    rgb=np.reshape(rgb2d,[h,w,3])
    return rgb
      


########################################

        
def deconstruct(data,piece,pad):
      
      width = data.shape[0]
      height = data.shape[1]
      if piece == 0:
          return data[0:int(width/2 + pad), 0:int(height/2 + pad),:]
      if piece == 1:
          return data[int(width/2 - pad):, 0:int(height/2 + pad),:]
      if piece == 2:
          return data[0:int(width/2 + pad), int(height/2 - pad):,:]
      if piece == 3:
          return data[int(width/2 - pad):, int(height/2 - pad):,:]
    
def construct( datas,shape):
      
      width = shape[0]
      height = shape[1]
      res = np.zeros(shape)
      res[0:int(width/2) , 0:int(height/2) ,:] = datas[0]
      res[int(width/2):, 0:int(height/2) ,:] = datas[1]
      res[0:int(width/2) , int(height/2):,:] = datas[2]
      res[int(width/2):, int(height/2):,:] = datas[3]
      
      return res



  
              
def geo_preprocess(data):
      out_data = []
    
      for k in range(4):
          out_data.append(np.rot90(data,k,(0,1)))
     
      for m in range(2):
          out_data.append(np.fliplr(out_data[m]))
          out_data.append(np.flipud(out_data[m]))
    
      return out_data
  
def geo_postprocess(data):
      out_data = []
      for k in range(len(data)):
          if k < 4 : 
              out_data.append(np.rot90(data[k],4-k,(0,1)))
          else:
              if k == 4:
                  out_data.append(np.fliplr(data[k]))
              elif k==5:
                  out_data.append(np.flipud(data[k]))
              elif k==6:
                  out_data.append(np.rot90(np.fliplr(data[k]),3,(0,1)))
              else:
                  out_data.append(np.rot90(np.flipud(data[k]),3,(0,1)))
    
    
      return np.mean(np.asarray(out_data),0)
    
def single_geo_preprocess(data,n):
   
      if n < 4:
          return np.rot90(data,n,(0,1))
      elif n == 4:
          return np.fliplr(data)
      elif n == 5:
          return np.flipud(data)
      elif n == 6:
          return np.fliplr(np.rot90(data,1,(0,1)))
      elif n == 7:
          return np.flipud(np.rot90(data,1,(0,1)))

      else:
          return None
  
def single_geo_postprocess(data,n):
      
      if n < 4:
          return np.rot90(data,4-n,(0,1))
      elif n == 4:
          return np.fliplr(data)
      elif n == 5:
          return np.flipud(data)
      elif n == 6:
          return (np.rot90(np.fliplr(data),3,(0,1)))
      elif n == 7:
          return (np.rot90(np.flipud(data),3,(0,1)))

      else:
          return None


########################################
# EVALUATIONS
          
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def psnr_calculation(img1, img2, unt):
    scale = 2
    if unt:
        
        img1 = (255*img1).astype(np.uint8).squeeze()
        img2 = (255*img2).astype(np.uint8).squeeze()
        PIXEL_MAX = 255
    else:
        img1 = img1.astype(np.float32).squeeze()
        img2 = img2.astype(np.float32).squeeze()
        img2[img2 == 0] = 0.0001
    if len(img1.shape) == 2:
        img1 = img1[scale:-scale,scale:-scale]
        img2 = img2[scale:-scale,scale:-scale]
    else:
        img1 = img1[scale:-scale,scale:-scale,:]
        img2 = img2[scale:-scale,scale:-scale,:]
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    
    t = np.sqrt(mse)
    
    aae = np.mean(np.sqrt((img1 - img2) ** 2))
    temp = np.mean(np.sqrt((img1 - img2) ** 2)/img2)
    
    return t, t/np.mean(img2),aae, temp



def evaluation(output, label, rad, FLAGS):
    # Evaluates PSNR of the images rendered by real and predicted parameters
    sens_predicted= np.transpose(np.reshape(output,[3,FLAGS.c_dim]))
    sens_actual =  label 
    
    rad=rad/np.amax(rad)
    h,w,numbands=rad.shape
            
    img_predicted=hyper_to_rgb(rad,h,w,numbands,(sens_predicted))
    img_real = hyper_to_rgb(rad,h,w,numbands, (sens_actual))
 
    psnr_value = psnr( img_predicted, img_real) 
    loss = np.mean(  ( (img_predicted-img_real)**2 )  )
    ssim_ = ssim(img_predicted, img_real, data_range=img_predicted.max() - img_predicted.min() ,multichannel=True)
    mrae_ = mrae(img_real,  img_predicted )
    return psnr_value , loss , ssim_, mrae_


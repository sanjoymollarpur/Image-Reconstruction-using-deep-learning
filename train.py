import sklearn
from sklearn import metrics
import sys
import glob
import re
import os
import time
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
from numpy import load
import tensorflow as tf 
import cv2
from scipy.ndimage import rotate 

import model as PAT


import subprocess
import time

def get_gpu_temperature():
    try:
        # Run the nvidia-smi command to get GPU information
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)

        # Extract the temperature value
        temperature = float(result.stdout.strip())
        return temperature

    except Exception as e:
        print(f"Error getting GPU temperature: {e}")
        return None

def check_temperature():
    current_temperature = get_gpu_temperature()
    print("current temp: ",current_temperature)
    if current_temperature is not None:
        safe_temperature_threshold = 60.0
        

        if current_temperature > safe_temperature_threshold:
            print(f"GPU Temperature ({current_temperature} C) exceeds safe threshold. Stopping the training.")
            # Add code to stop or pause the training process here
            time.sleep(1200)

#while True:
 #   check_temperature()
  #  time.sleep(300)  # 300 seconds = 5 minutes



def normalize_data(X, axis = 0):
    mu = np.mean(X,axis =0)
    sigma = np.std(X, axis=0)

    X_norm = 2*(X-mu)/sigma
    return np.nan_to_num(X_norm)




def augmentation(image, imageB, org_width=160,org_height=224, width=190, height=262):
    max_angle=20
    image=cv2.resize(image,(height,width))
    imageB=cv2.resize(imageB,(height,width))

    angle=np.random.randint(max_angle)
    if np.random.randint(2):
        angle=-angle
    
    image=rotate(image,angle)
    imageB=rotate(imageB,angle)

    # image=image.rotate(angle,resize=True)
    # imageB=imageB.rotate(angle,resize=True)

    xstart=np.random.randint(width-org_width)
    ystart=np.random.randint(height-org_height)
    image=image[xstart:xstart+org_width,ystart:ystart+org_height]
    imageB=imageB[xstart:xstart+org_width,ystart:ystart+org_height]

    if np.random.randint(2):
        image=cv2.flip(image,1)
        imageB=cv2.flip(imageB,1)

    if np.random.randint(2):
        image=cv2.flip(image,0)
        imageB=cv2.flip(imageB,0)

    image=cv2.resize(image,(org_height,org_width))
    imageB=cv2.resize(imageB,(org_height,org_width))

    return image,imageB




def split_train_test(idc, test_fraction):

    random.seed(0) # deterministic
    random.shuffle(idc) # in-place
    tmp = int((1-test_fraction)*len(idc))
    idc_train = idc[:tmp] # this makes a copy
    idc_test = idc[tmp:]
    return idc_train, idc_test



print("hello")

# w_artifact_file = h5py.File('/mnt/sdb1/dataset/data/mice_sparse32_recon.mat', 'r')
w_artifact_file = h5py.File('dataset/data/mice_sparse32_recon.mat', 'r')
variables_w = w_artifact_file.items()
print(variables_w)
print("hello")
for var in variables_w:
    name_w_artifact = var[0]
    data_w_artifact = var[1]
    print(data_w_artifact.shape, name_w_artifact)
    if type(data_w_artifact) is h5py.Dataset:
        w_artifact = data_w_artifact
        w_artifact=normalize_data(w_artifact)




wo_artifact_file = h5py.File('dataset/data/mice_full_recon.mat', 'r')

variables_wo = wo_artifact_file.items()

for var in variables_wo:
    name_wo_artifact = var[0]
    data_wo_artifact = var[1]
    if type(data_wo_artifact) is h5py.Dataset:
        wo_artifact = data_wo_artifact # NumPy ndArray / Value
        wo_artifact=normalize_data(wo_artifact)



for i in range(0,278):
    i1, i2 = augmentation(w_artifact[i], wo_artifact[i])
    print(i1.shape)


colormap = 'gray'
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
plt.subplot(1, 2, 1)
plt.imshow(w_artifact[3], cmap=colormap)  # Use an appropriate colormap


plt.savefig('gt.png', format='png', bbox_inches='tight')
plt.title('GT')
# plt.colorbar()  # Add a colorbar for intensity scale
# plt.show()

plt.subplot(1, 2, 2)
plt.imshow(wo_artifact[3], cmap=colormap)
plt.savefig('new/arti.png', format='png', bbox_inches='tight')
plt.title('arti')
#plt.show()
plt.close()




print("hello", w_artifact.shape, wo_artifact.shape)


test_fraction=0.2
idc_train, idc_test = split_train_test(list(range(w_artifact.shape[0])), test_fraction)

#print(idc_train)
#print(idc_test)

#for b in idc_train:
#    print(b)
batchsize=1

model = PAT.unet(input_size=(512, 512, 1), learning_rate=0.001)  # Assuming your unet function is defined
#print(model.summary())


e=0
loss1=float("inf")
for e in range(0,200):
    print(f"epoch {e+1}/100")
    loss=0
    
    for b in range(0, len(idc_train), batchsize):
        idcMB_train = list(idc_train[b : b+batchsize])
        print(idcMB_train)
        X_train = np.expand_dims(np.require(w_artifact[idcMB_train, : ,:], dtype = np.float32), axis = 0)


        Y_train = np.require(w_artifact[idcMB_train, :, :] - wo_artifact[idcMB_train, :, :], dtype = np.float32)
        X_train = tf.reshape(X_train, (1, 512, 512, 1))
        Y_train = tf.reshape(Y_train, (1, 512, 512, 1))
        
        print(e, b, X_train.shape, Y_train.shape)

        model.train_on_batch(X_train,Y_train)
        
        
        check_temperature()
        if (e+1)%20 == 0:
            
            for l in range(0, len(idc_test), batchsize):
                # print("test data, epoch ",l,e)
                idcMB_test = list(idc_test[l : l+batchsize])

                X_test = np.expand_dims(np.require(w_artifact[idcMB_test, : ,:], dtype = np.float32), axis = 0)

                Y_test = np.require(w_artifact[idcMB_test, : ,:] - wo_artifact[idcMB_test, :, :], dtype = np.float32)

                X_test = tf.reshape(X_test, (1, 512, 512, 1))
                Y_test = tf.reshape(Y_test, (1, 512, 512, 1))

                predictions = model.predict(X_test)


                colormap = 'gray'
                plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
                plt.subplot(1, 3, 1)
                plt.imshow(X_test[0], cmap=colormap)  # Use an appropriate colormap
                plt.title('Artifactual')
                
                plt.subplot(1, 3, 2)
                plt.imshow(X_test[0]-Y_test[0], cmap=colormap)
                plt.title('Ground Truth')


                plt.subplot(1, 3, 3)
                plt.imshow(X_test[0] - predictions[0], cmap=colormap)
                plt.title('pred')
                plt.savefig(f'/mnt/sdb1/pred/pred-{l}-{e}.png', format='png', bbox_inches='tight')
               
                # plt.show()
                plt.close()




                
                loss+= model.evaluate(X_test, Y_test)
        
            if loss<loss1:
                loss1=loss
                model.save(f"/mnt/sdb1/new/save_model/model{e}.h5")

model.save(f"/mnt/sdb1/new/save_model/model_end.h5")
        











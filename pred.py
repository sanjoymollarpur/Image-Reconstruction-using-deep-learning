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
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

import model as PAT


import subprocess
import time


def normalize_data(X, axis = 0):
    mu = np.mean(X,axis =0)
    sigma = np.std(X, axis=0)

    X_norm = 2*(X-mu)/sigma
    return np.nan_to_num(X_norm)

from scipy.io import loadmat

filepath_gt = 'dataForDemo/GT.mat'
filepath_pred = 'dataForDemo/predicted.mat'
filepath_arti = 'dataForDemo/artifactual.mat'



from scipy.io import loadmat

a = loadmat(filepath_gt)
pred = loadmat(filepath_pred)
arti = loadmat(filepath_arti)

# print(a)

model = tf.keras.models.load_model('new/save_model/model_end.h5')

a=a["GT"]
pred=pred["predicted"]
arti=arti["artifactual"]
row=len(a)
print(row,len(a[0]), a[0])

colormap = 'jet'
colormap = 'gray'

artifactual1=[]
gt1=[]
paper_pred=[]
my_pred=[]

for i in range(0,len(a)):
    plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
    plt.subplot(1, 3, 1)
    plt.imshow(a[i], cmap=colormap)  # Use an appropriate colormap
    plt.title('GT')


    # plt.savefig('gt.png', format='png', bbox_inches='tight')
    
    # plt.colorbar()  # Add a colorbar for intensity scale
    # plt.show()

    plt.subplot(1, 3, 2)
    plt.imshow(pred[i], cmap=colormap)
    plt.title('pred')
    # plt.savefig('pred.png', format='png', bbox_inches='tight')
    
    # plt.show()

    plt.subplot(1, 3, 3)
    plt.imshow(arti[i], cmap=colormap)
    plt.title('artifactual')
    # plt.savefig(f'pred_img/arti-{i}.png', format='png', bbox_inches='tight')
    
    # plt.show()
    plt.close()


    arti[i]=normalize_data(arti[i])
    a[i]=normalize_data(a[i])
    X_test = np.expand_dims(np.require(arti[i], dtype = np.float32), axis = 0)
    # X_test = np.expand_dims(np.require(wo_artifact[idcMB_test, : ,:], dtype = np.float32), axis = 0)

    Y_test = np.require(arti[i] - a[i], dtype = np.float32)
    # Y_test = np.require(wo_artifact[idcMB_test, : ,:], dtype = np.float32)

    X_test = tf.reshape(X_test, (1, 512, 512, 1))
    Y_test = tf.reshape(Y_test, (1, 512, 512, 1))

    predictions = model.predict(X_test)


    colormap = 'gray'
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
    plt.subplot(1, 4, 1)
    plt.imshow(X_test[0]-Y_test[0], cmap=colormap)  # Use an appropriate colormap
    plt.title('GT')
    
    plt.subplot(1, 4, 2)
    plt.imshow(X_test[0], cmap=colormap)
    plt.title('Artifactual')


    plt.subplot(1, 4, 3)
    plt.imshow(X_test[0]-predictions[0], cmap=colormap)
    plt.title('My prediction result')
    plt.subplot(1, 4, 4)
    plt.imshow(pred[i], cmap=colormap)
    plt.title('Paper prediction result')
    artifactual1.append(arti[i])
    gt1.append(X_test[0]-Y_test[0])
    paper_pred.append(pred[i])
    my_pred.append(X_test[0]-predictions[0])
    # plt.savefig(f'test/my/pred-{l}.png', format='png', bbox_inches='tight')
    # plt.savefig(f'dataForDemo/compare/{i}.png', format='png', bbox_inches='tight')

    
    
    plt.show()
    plt.close()

# saving_path="dataForDemo/pred_mat"
# sio.savemat(saving_path + '/artifactual.mat', {"artifactual" : artifactual1})
# sio.savemat(saving_path + '/GT.mat', {"GT" : gt1})
# sio.savemat(saving_path + '/paper_prediction.mat', {"paper prediction" : paper_pred})
# sio.savemat(saving_path + '/my_prediction.mat', {"my prediction" : my_pred})
        











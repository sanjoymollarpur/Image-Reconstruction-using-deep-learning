import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# data/9250634/mice_sparse4_recon.mat
# filepath = 'test_GT.mat'
filepath_gt = 'result/GT.mat'
filepath_pred = 'result/predicted.mat'
filepath_arti = 'result/artifactual.mat'



from scipy.io import loadmat

a = loadmat(filepath_gt)
pred = loadmat(filepath_pred)
arti = loadmat(filepath_arti)

# print(a)

a=a["GT"]
pred=pred["predicted"]
arti=arti["artifactual"]
row=len(a)
print(row,len(a[0]), a[0])

colormap = 'jet'
colormap = 'gray'
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
    plt.savefig(f'pred_img/arti-{i}.png', format='png', bbox_inches='tight')
    
    plt.show()
    plt.close()

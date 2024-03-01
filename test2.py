from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

split_paths = sorted(glob('/home/dserrano/Documents/datasets/lsmi_mask/sony/valid/*_G_AS.png'))

for path in split_paths:
    img = Image.open(path)
    img = np.array(img)
    plt.imsave(f'/home/dserrano/Documents/datasets/lsmi_mask/sony/valid/{path.split("/")[-1].split("_")[0]}_{path.split("/")[-1].split("_")[1]}.png', img)

# img1 = Image.open('/home/dserrano/Downloads/lightroom-tutorial-1.png').convert('RGB')
# img1 = np.array(img1)[206:918, 1395:1818]
# plt.imsave('/home/dserrano/Downloads/lightroom-tutorial-1_cropped.png', img1)
#
# img2 = Image.open('/home/dserrano/Downloads/lightroom-tutorial-2.png').convert('RGB')
# img2 = np.array(img2)[206:918, 1395:1818]
# plt.imsave('/home/dserrano/Downloads/lightroom-tutorial-2_cropped.png', img2)
#
# img3 = Image.open('/home/dserrano/Downloads/lightroom-tutorial-3.png').convert('RGB')
# img3 = np.array(img3)[206:918, 1395:1818]
# plt.imsave('/home/dserrano/Downloads/lightroom-tutorial-3_cropped.png', img3)
#
# img4 = Image.open('/home/dserrano/Downloads/lightroom-tutorial-4.png').convert('RGB')
# img4 = np.array(img4)[206:918, 1395:1818]
# plt.imsave('/home/dserrano/Downloads/lightroom-tutorial-4_cropped.png', img4)

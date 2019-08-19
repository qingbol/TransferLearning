import os,sys
import h5py
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
import seaborn as sns
import math
# %matplotlib inline
from tqdm import tqdm
from PIL import Image
from  scipy import ndimage

def augmentation():
    data_root='./data_bully'
    src_train_dir=os.path.join(data_root,'train_data_09/')
    dest_train_dir=os.path.join(data_root,'train_data_aug')
    # print(src_train_dir)
    # print(dest_train_dir)

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')


    for root, dirs, files in os.walk(src_train_dir):
        path = root.split(os.sep)
    #     print(path)
    #     print(dirs)
    #     print(files)
        cur_dir=os.path.basename(root)
        print(os.path.basename(root))
        #skip the root directory
        if cur_dir=="":
            continue
        src_lab_dir=os.path.join(src_train_dir,cur_dir)
        dest_lab_dir=os.path.join(dest_train_dir,cur_dir)
        if not os.path.exists(dest_lab_dir):
            os.makedirs(dest_lab_dir)


    #     root_depth = len(root.split(os.path.sep))
    #     print(root_depth)

        class_size=9478
        file_count=len(files)
    #     print(file_count)
        #nb of generations per image for this class label in order to make it size ~= class_size
        ratio=math.floor(class_size/file_count)-1
        print(file_count,ratio, file_count*(ratio+1))
        if ratio<1:
            continue
        
        for file in files:
            img=load_img(os.path.join(src_lab_dir,file))
            x=img_to_array(img)
    #         print(x)
            x=x.reshape((1,) + x.shape)
            i=0
            for batch in datagen.flow(x, batch_size=1,save_to_dir=dest_lab_dir, save_format='jpg'):
                i+=1
                if i > ratio:
                    break 

if __name__ == '__main__':
    augmentation()
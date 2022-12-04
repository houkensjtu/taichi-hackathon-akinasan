import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
from PIL import Image

def RGB2GRAY(img):
    temp_img = np.zeros([28,28])
    temp_img = img[:,:,0]*0.3+img[:,:,1]*0.59+img[:,:,2]*0.11
    return temp_img

def process_img(path,key):
    image_array = Image.open(path)
    new_img = image_array.resize((28, 28))
    np_array = np.array(new_img)
    gray_img = RGB2GRAY(np_array)
    gray_img = np.expand_dims(gray_img, axis=0)

    label_array = [0., 0., 0., 0., 0.]
    if key == 2:
        label_array = [0., 0., 1., 0., 0.]
    elif key == 3:
        label_array = [0., 0., 0., 1., 0.]
    elif key == 0:
        label_array = [1., 0., 0., 0., 0.]
    elif key == 1:
        label_array = [0., 1., 0., 0., 0.]
    elif key == 4:
        label_array = [0., 0., 0., 0., 1.]
    return (gray_img, label_array)

if __name__ == '__main__':
    train_labels = np.zeros((1, 5), 'float')
    train_imgs = np.zeros([1, 28, 28])
    path = 'training_data_office'
    files = os.listdir(path)

    for i in files:
        sub_path = path + '/' + i
        sub_imgs = os.listdir(sub_path)
        for j in sub_imgs:
            img_path = sub_path + '/' + j
            key = int(j[0])
            image_arr, label_arr = process_img(img_path, key)
            train_imgs = np.vstack((train_imgs, image_arr))
            train_labels = np.vstack((train_labels, label_arr))

    train_imgs = train_imgs[1:, :]
    train_labels = train_labels[1:, :]

    print('-----------------')
    print(train_labels)
    print(train_imgs.shape)
    print('-----------------')

    file_name = str(int(time.time()))
    directory = "training_data_npz"

    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        np.savez(directory + '/' + file_name + '.npz', train_imgs=train_imgs, train_labels=train_labels)
    except IOError as e:
        print(e)
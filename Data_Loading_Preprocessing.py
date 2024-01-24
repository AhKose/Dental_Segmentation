import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zipfile
import glob
from PIL import Image
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(image_folder, mask_folder, size=(256, 256), color_mode='grayscale'):
    images = []
    masks = []
    image_files = sorted(glob.glob(image_folder + '/*.JPG')) + sorted(glob.glob(image_folder + '/*.jpg'))
    mask_files = sorted(glob.glob(mask_folder + '/*.jpg'))

    print(f"Found {len(image_files)} image files and {len(mask_files)} mask files.")

    for image_filename in image_files:
        basename = os.path.basename(image_filename)
        # Change the extension to lowercase to match the mask files
        basename = os.path.splitext(basename)[0] + '.jpg'
        mask_filename = os.path.join(mask_folder, basename)

        if os.path.exists(mask_filename):
            img = Image.open(image_filename)
            mask = Image.open(mask_filename)

            if color_mode == 'grayscale':
                img = img.convert('L')
                mask = mask.convert('L')

            img = img.resize(size)
            mask = mask.resize(size)

            img = np.array(img) / 255.0
            mask = np.array(mask) / 255.0

            if color_mode == 'grayscale':
                img = np.expand_dims(img, axis=-1)
                mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)
        else:
            print(f"Missing mask for image: {image_filename}")

    return np.array(images), np.array(masks)

# Paths to data folders
image_folder = "/content/drive/MyDrive/Teeth/Radiographs"
mask_folder = "/content/drive/MyDrive/Teeth/teeth_mask"

images, masks = load_images_from_folder(image_folder, mask_folder)
if len(images) > 0 and len(masks) > 0:
    images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.2)
else:
    print("No matching image/mask pairs found.")

import os
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_folder(image_folder, mask_folder, size=(256, 256), color_mode='grayscale'):
    """
    Load images and corresponding masks from the given folders, resize them,
    convert them to grayscale if specified, and normalize the pixel values.

    Parameters:
    image_folder (str): Path to the folder containing images.
    mask_folder (str): Path to the folder containing corresponding masks.
    size (tuple): Target size for resizing the images and masks.
    color_mode (str): If 'grayscale', convert images and masks to grayscale.

    Returns:
    tuple: A tuple containing two numpy arrays: one for images and one for masks.
    """
    images = []
    masks = []
    image_files = sorted(glob.glob(image_folder + '/*.JPG')) + sorted(glob.glob(image_folder + '/*.jpg'))
    mask_files = sorted(glob.glob(mask_folder + '/*.jpg'))

    print(f"Found {len(image_files)} image files and {len(mask_files)} mask files.")

    for image_filename in image_files:
        basename = os.path.basename(image_filename)
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

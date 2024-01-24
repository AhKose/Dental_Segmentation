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
# Define data generators with augmentation parameters
def create_augmented_data_generator():
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    return image_datagen, mask_datagen

# Apply the same transformations to images and masks
def generate_augmented_data(images, masks, batch_size):
    # Initialize data generators
    image_datagen, mask_datagen = create_augmented_data_generator()

    # Provide the same seed to image_datagen and mask_datagen to ensure the transformations for image and mask are the same
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow(images, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        yield (img, mask)

# Usage example
batch_size = 16
augmented_data_generator = generate_augmented_data(images_train, masks_train, batch_size)
def conv_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def unet(input_size=(256, 256, 1), num_filters=64, dropout=0.5):
    inputs = Input(input_size)

    # Contracting Path
    c1 = conv_block(inputs, num_filters)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv_block(p1, num_filters * 2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv_block(p2, num_filters * 4)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv_block(p3, num_filters * 8)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = conv_block(p4, num_filters * 16)

    # Expansive Path
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv_block(u6, num_filters * 8)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv_block(u7, num_filters * 4)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv_block(u8, num_filters * 2)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv_block(u9, num_filters)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = unet()
model.fit(augmented_data_generator, steps_per_epoch=len(images_train) // batch_size, epochs=10)
import matplotlib.pyplot as plt

# Function to display images in a grid
def plot_images(images_arr, titles_arr=None, figsize=(20, 10), rows=1):
    fig, axes = plt.subplots(rows, len(images_arr) // rows, figsize=figsize)
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        if img.shape[-1] == 1:  # If single channel image, display as grayscale
            ax.imshow(np.squeeze(img), cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
    if titles_arr:
        for ax, title in zip(axes, titles_arr):
            ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Predict on test images
predictions = model.predict(images_test, batch_size=batch_size)

# Select a random sample from test images
num_samples = 3  # Number of samples you want to visualize
indices = np.random.choice(range(len(images_test)), num_samples, replace=False)

sample_images = images_test[indices]
sample_masks = masks_test[indices]
sample_predictions = predictions[indices]

# Plotting original images, true masks and predicted masks
for i in range(num_samples):
    plot_images([sample_images[i], sample_masks[i], sample_predictions[i]],
                titles_arr=['Original Image', 'True Mask', 'Predicted Mask'],
                figsize=(20, 5),
                rows=1)

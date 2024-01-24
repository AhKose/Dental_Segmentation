
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

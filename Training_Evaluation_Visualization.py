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

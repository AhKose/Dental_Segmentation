import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_images_from_folder
from model_1 import model #To use model_2 change the import statement

class DiceCoefficient(Metric):
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        sum_ = K.sum(y_true_f) + K.sum(y_pred_f)
        self.dice.assign(2. * intersection / (sum_ + K.epsilon()))

    def result(self):
        return self.dice

# Paths to data folders, replace it with your paths
image_folder = "/content/Teeth/Radiographs"
mask_folder = "/content/Teeth/teeth_mask"

try:
    images, masks = load_images_from_folder(image_folder, mask_folder)
    images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size=0.2)

except Exception as e:
    print(f"Error loading images: {e}")

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', DiceCoefficient()])

try:
  history = model.fit(images_train, masks_train, batch_size=16, epochs=50, validation_split=0.1, shuffle=True, callbacks=[early_stopping])
except RuntimeError as e:
    print(f"Runtime error during model training: {e}")
except Exception as e:
    print(f"An error occurred during model training: {e}")

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
predictions = model.predict(images_test, batch_size=16)

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

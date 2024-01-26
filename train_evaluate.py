import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K

model.fit(augmented_data_generator, steps_per_epoch=len(images_train) // batch_size, epochs=10)

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

# Use it in your model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', DiceCoefficient()])
history = model.fit(images_train, masks_train, batch_size=16, epochs=20, validation_split=0.1, shuffle=True)


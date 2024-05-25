import os
import zipfile

import numpy as np

# Convert or Extract Zip File

# local_zip = 'C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\archive (1).zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning')
# zip_ref.close()

base_dir = "C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN\\horse-or-human"
validation_dir = os.path.join(base_dir, 'validation')
# directory with our training horse pictures
train_horse_dir = os.path.join(
    'C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN\\horse-or-human\\train\\horses')

# directory with our training human pictures
train_human_dir = os.path.join(
    "C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN\\horse-or-human\\train\\humans")

validation_horse_dir = os.path.join(
    validation_dir, 'horses')

validation_human_dir = os.path.join(
    validation_dir, 'humans')

# Names of traing horse images
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

# Names of Training Human Images
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

# calculating Number of images both horses and humans training set holds
print("Total Training Horse Image: ", len(train_horse_names))
print("Total Training Human Image: ", len(train_human_names))

print("Total Validation Horse Image: ", len(os.listdir(validation_horse_dir)))
print("Total Validation Human Image: ", len(os.listdir(validation_human_dir)))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration, so we'll specify that here.
n_cols = 4
n_rows = 4

# index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(n_cols * 4, n_rows * 4)
pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index - 8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index - 8:pic_index]]
for i, img_path in enumerate(next_horse_pix + next_human_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(n_rows, n_cols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other (
    # 'humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
# updated to do image augmentation
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 128 using train_datagen generator.
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN\\horse-or-human',
    # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=128,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    'C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN\\horse-or-human\\validation',
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=8,  # 2000 images = batch_size * steps
    epochs=15,
    verbose=2,
    validation_steps=50,  # 1000 images = batch_size * steps
    callbacks=[callbacks])

import numpy as np
from keras.preprocessing import image

path = 'horse.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0:
    print("is a human")
else:
    print("is a horse")

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

Successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=Successive_outputs)

human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
img_path = random.choice(human_img_files + horse_img_files)
img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
# Rescale by 1/255
x /= 255
# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)
# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]
# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        # Just do this for the conv / max pool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map
        # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[1]
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size:(i + 1) * size] = x
        # Display the grid
        scale = 20. / n_features

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

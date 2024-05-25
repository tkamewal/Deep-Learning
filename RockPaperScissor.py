import os
import zipfile

# local_zip = "C:\\Users\\TANMAY KAMEWAL\\Downloads\\rps.zip"
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall("C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN")
# zip_ref.close()

base_dir = "C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\CNN"

train_dir = os.path.join(base_dir, 'rps')

train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')

train_rock_files = os.listdir(train_rock_dir)
train_paper_files = os.listdir(train_paper_dir)
train_scissors_files = os.listdir(train_scissors_dir)

print(train_rock_files[:10])
print(train_paper_files[:10])
print(train_scissors_files[:10])

print("Total Rock images: ",len(train_rock_files))
print("Total Paper images: ", len(train_paper_files))
print("Total Scissors images: ", len(train_scissors_files))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

n_rows = 4
n_cols = 4
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(n_cols * 4, n_rows * 4)
pic_index += 8

# Assuming there are 4 rows and 4 columns, so the total number of subplots is 16.
total_subplots = n_rows * n_cols

next_rock_pix = [os.path.join(train_rock_dir, fname) for fname in train_rock_files[pic_index - 8:pic_index]]
next_paper_pix = [os.path.join(train_paper_dir, fname) for fname in train_paper_files[pic_index - 8:pic_index]]
next_scissors_pix = [os.path.join(train_scissors_dir, fname) for fname in train_scissors_files[pic_index - 8:pic_index]]

# Concatenate the image lists
all_images = next_rock_pix + next_paper_pix + next_scissors_pix

# Loop over the images up to the total number of subplots
for i, img_path in enumerate(all_images[:total_subplots]):
    sp = plt.subplot(n_rows, n_cols, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # the image Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = callback()
history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=30,
    verbose=1,
    callbacks=[callbacks]
)


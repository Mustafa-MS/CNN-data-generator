import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
import pandas

nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")

base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"
all_image_paths = os.listdir(base_dir)

nodules = nodules_csv.rename(columns = {'SN':'ID'})
labels= nodules.iloc[:,1]
labels = labels.to_numpy()

def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])

    # The last subplot has no image because there are only 19 images.
    plt.show()

class DataGenerator(Sequence):
# Learned from https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
  def __init__(self, all_image_paths, labels, base_dir, output_size, shuffle=False, batch_size=10):
    """
    Initializes a data generator object
      :param csv_file: file in which image names and numeric labels are stored
      :param base_dir: the directory in which all images are stored
      :param output_size: image output size after preprocessing
      :param shuffle: shuffle the data after each epoch
      :param batch_size: The size of each batch returned by __getitem__
    """
    self.imgs = all_image_paths
    self.base_dir = base_dir
    self.output_size = output_size
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.labels = labels
    self.on_epoch_end()

  def on_epoch_end(self):
    self.indices = np.arange(len(self.imgs))
    if self.shuffle:
      np.random.shuffle(self.indices)

  def __len__(self):
    return int(len(self.imgs) / self.batch_size)

  def __getitem__(self, idx):
    ## Initializing Batch
    #  that one in the shape is just for a one channel images
    # if you want to use colored images you might want to set that to 3
    X = np.empty((self.batch_size, *self.output_size,1))
    # (x, y, h, w)
    y = np.empty((self.batch_size, 5, 1))

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

    for i, data_index in enumerate(indices):
      img_path = os.path.join(self.base_dir,
                  self.imgs[data_index])

      img = np.load(img_path)
      #plot_nodule(img)
      img = np.expand_dims(img, axis=3)
      ## this is where you preprocess the image
      ## make sure to resize it to be self.output_size

      label = self.labels[data_index]
      ## if you have any preprocessing for
      ## the labels too do it here

      X[i,] = img
      y[i] = label

    return X, y


## Defining and training the model

model = Sequential([
  ## define the model's architecture
    layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=128, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=256, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling3D(),
    layers.Dense(units=512, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(units=1, activation="sigmoid"),
])

train_gen = DataGenerator(all_image_paths, labels, base_dir, (31, 31, 31), batch_size=64, shuffle=True)

## compile the model first of course

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# now let's train the model
model.fit(train_gen, epochs=5)

# Import the packages we need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def getCatsAndDogsData():
  # Download and extract the Data Set
  zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                                    fname="cats_and_dogs_filtered.zip", extract=True)

  # Grab the location of the unzipped data
  base_dir, _ = os.path.splitext(zip_file)

  # Define the path to the Training and Validation Datasets
  train_dir = os.path.join(base_dir, 'train')
  validation_dir = os.path.join(base_dir, 'validation')

  return train_dir, validation_dir

def getTrainingDirs(train_dir):
  # Directory with our training cat pictures
  train_cats_dir = os.path.join(train_dir, 'cats')
  print ('Total training cat images:', len(os.listdir(train_cats_dir)))

  # Directory with our training dog pictures
  train_dogs_dir = os.path.join(train_dir, 'dogs')
  print ('Total training dog images:', len(os.listdir(train_dogs_dir)))

  return train_cats_dir, train_dogs_dir

def getValidationDirs(validation_dir):
   # Directory with our validation cat pictures
  validation_cats_dir = os.path.join(validation_dir, 'cats')
  print ('Total validation cat images:', len(os.listdir(validation_cats_dir)))

  # Directory with our validation dog pictures
  validation_dogs_dir = os.path.join(validation_dir, 'dogs')
  print ('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

  return validation_cats_dir, validation_dogs_dir

def getCatsAndDogsImageNames(cats_dir, dogs_dir):
  train_cats_names = os.listdir(cats_dir)
  train_dogs_names = os.listdir(dogs_dir)

  return train_cats_names, train_dogs_names


def showImageGrid(image_dir, num_rows=2, num_cols=4):  
  image_labels = os.listdir(image_dir)
  num_pix = num_rows * num_cols
  # Index for iterating over images
  pic_index = 0
  # Set up matplotlib fig, and size it to fit 4x4 pics
  fig = plt.gcf()
  fig.set_size_inches(num_cols * 4, num_rows * 4)

  pic_index += num_pix
  next_pix = [os.path.join(image_dir, fname) 
                  for fname in image_labels[pic_index-num_pix:pic_index]]
  
  for i, img_path in enumerate(next_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(num_rows, num_cols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

  plt.show()

def printLossAndAccuracy(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,max(plt.ylim())])
  plt.title('Training and Validation Loss')
  plt.show()

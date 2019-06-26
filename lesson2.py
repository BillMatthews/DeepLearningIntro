import os

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getData():
  data_file = keras.utils.get_file(fname="auto-mpg.data", origin="https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
  # The data has the following coluns
  column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin Country']
  # Read in the file into a Pandas Dataset
  raw_dataset = pd.read_csv(data_file, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
  # We will discard any rows that contain data with missing values
  raw_dataset = raw_dataset.dropna()
  return raw_dataset

def preprocessCategoricalData(data):
    # Use One-Hot-Encoding for categorical data
    origin = data.pop('Origin Country')
    data['USA'] = (origin == 1)*1.0
    data['Europe'] = (origin == 2)*1.0
    data['Japan'] = (origin == 3)*1.0
    return data
    

def splitDataset(dataset):
  # Split the data into Train and Test Sets
  # We randomly select 80% of the records for our Training Dataset
  train_dataset = dataset.sample(frac=0.8,random_state=0)
  # We then use the remaining records as our Test Dataset
  test_dataset = dataset.drop(train_dataset.index)

  # Extract the labels for training and testing
  train_labels = train_dataset.pop('MPG')
  test_labels = test_dataset.pop('MPG')

  return train_dataset, train_labels, test_dataset, test_labels

def displayPairPlots(data, labels):
  dataset = data.copy()
  dataset['MPG'] = labels
  sns.pairplot(dataset, diag_kind="kde")

def norm(x, train_stats):
  return (x - train_stats['mean']) / train_stats['std']

def normaliseData(train_dataset, test_dataset):
  train_stats = train_dataset.describe()
  train_stats = train_stats.transpose() 

  normed_train_data = norm(train_dataset, train_stats)
  normed_test_data = norm(test_dataset, train_stats)

  return normed_train_data, normed_test_data

def plotHistory(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

def displayScatter(expected, predictions):
  plt.scatter(expected, predictions)
  plt.xlabel('True Values [MPG]')
  plt.ylabel('Predictions [MPG]')
  plt.axis('equal')
  plt.axis('square')
  plt.xlim([0,plt.xlim()[1]])
  plt.ylim([0,plt.ylim()[1]])
  _ = plt.plot([-100, 100], [-100, 100])
  plt.show()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


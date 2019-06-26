# Import the packages we need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def printSampleImages(image_data, image_labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_data[i], cmap=plt.cm.binary)
        plt.xlabel(image_labels[i])
    plt.show()

def displayLossAndAccuracy(history):
  plt.plot(history.history['loss'], color='red', label='Loss')
  plt.plot(history.history['acc'], color='blue', label='Accuracy')
  plt.legend()
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.show()

def printSingleImage(image_data, image_labels, image_index, title=""):
  plt.imshow(image_data[image_index], cmap=plt.cm.binary)
  plt.xlabel(image_labels[image_index])
  if len(title) > 0 :
    plt.title(title)
  plt.show()

def printSampleIncorrectImages(image_data, image_labels, model):
    incorrects = np.nonzero(model.predict_classes(image_data).reshape((-1,)) != image_labels)
    
    plt.figure(figsize=(10,10))
    max_items = 25 if len(incorrects[0]) > 25 else len(incorrects[0])
    for i in range(max_items):
        incorrect_image_index = incorrects[0][i]
        incorrect_image_data = np.array([image_data[incorrect_image_index]])
        expected_class = image_labels[incorrect_image_index]
        predicted_class = model.predict_classes(incorrect_image_data)
        errorTitle = "Expected class: '%d'\nPredicted class: '%d'"%(expected_class, predicted_class)
    
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()
        plt.imshow(image_data[incorrect_image_index], cmap=plt.cm.binary)
        plt.xlabel(errorTitle)
    plt.show()

def displayConfusionMatrix(expected, predicted):
  import seaborn as sn
  from sklearn.metrics import confusion_matrix

  confusion_matrix = confusion_matrix(y_true = expected, y_pred = predicted)
  plt.figure(figsize=(12,8))
  ax = plt.subplot()
  sn.heatmap(confusion_matrix, annot=True, ax = ax)
  ax.set_xlabel('Predicted labels');
  ax.set_ylabel('Actual labels'); 
  ax.set_title('Confusion Matrix'); 

  plt.show()

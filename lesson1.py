# Import the packages we need
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def plotTrainingData(miles_travelled, expected_cost):
  # Here we are using MatPlotLib (which we imported as plt) to plot the graph for us.
  plt.plot(miles_travelled, expected_cost, 'o', color='red')
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.show()

def displayModelWeightsAndBias(model):
  for i, layer in enumerate(model.layers):
    print("Layer %d"%(i))
    print("\tWeights are: ", layer.get_weights()[0])
    print("\tBias is:" , layer.get_weights()[1])
    print("\n")

def plotTrainingVsModel(miles_travelled, expected_cost, model):
  predicted_costs = model.predict(miles_travelled)

  # Plot the data out
  plt.plot(miles_travelled, expected_cost, 'o', color='red', label="Expected Cost")
  plt.plot(miles_travelled, predicted_costs, 'x', color='blue', label="Predicted Cost")
  plt.title("Expected Cost vs Model Predicted Cost")
  plt.legend()
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.show()

def displayTrainingVsPredictedValues(miles_travelled, expected_cost, model):
  predicted_costs = model.predict(miles_travelled)
  for x_val, y_actual, y_pred in zip(miles_travelled, expected_cost, predicted_costs):
    print("(%1.3f, %1.3f)  -> \t (%1.3f, %1.3f)"%(x_val, y_actual, x_val, y_pred))

def displayConfusionMatrix(true_labels, predicted_labels):
  from sklearn.metrics import confusion_matrix
  import seaborn as sn

  confusion_matrix = confusion_matrix(y_true = true_labels, y_pred = predicted_labels)
  plt.figure(figsize=(12,8))
  ax = plt.subplot()
  sn.heatmap(confusion_matrix, annot=True, ax = ax)
  ax.set_xlabel('Predicted labels');
  ax.set_ylabel('Actual labels'); 
  ax.set_title('Confusion Matrix'); 

  plt.show()



def plotLoss(loss_history, from_epoch=0, to_epoch=0):
  if to_epoch == 0 or to_epoch >= len(loss_history):
    to_epoch = len(loss_history)
  
  x = range(from_epoch, to_epoch)
  plt.plot(x, loss_history[from_epoch:to_epoch])
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.title("Model loss between epochs %d and %d"%(from_epoch, to_epoch))
  plt.xlabel("Epoches")
  plt.ylabel("Model Loss")
  plt.show()




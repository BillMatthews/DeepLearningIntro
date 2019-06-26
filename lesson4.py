# Import the packages we need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

def showSampleImages(image_data, image_labels, class_labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_data[i], cmap=plt.cm.binary)
        plt.xlabel(class_labels[image_labels[i]])
    plt.show()

def printSampleCnnIncorrectImages(source_data, image_labels, class_names, model):
  incorrects = np.nonzero(model.predict_classes(source_data).reshape((-1,)) != image_labels)
  
  plt.figure(figsize=(10,10))
  max_items = 25 if len(incorrects[0]) > 25 else len(incorrects[0])
  
  for i in range(max_items):
    incorrect_image_index = incorrects[0][i]
    incorrect_image_data = np.array([source_data[incorrect_image_index]])
    expected_class = class_names[image_labels[incorrect_image_index]]
    preds = model.predict_classes(incorrect_image_data)    
    predicted_class = class_names[model.predict_classes(incorrect_image_data)[0]]
    errorTitle = "Expected class: '%s'\nPredicted class: '%s'"%(expected_class, predicted_class)

    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.tight_layout()
    #plt.imshow(image_data[incorrect_image_index], cmap=plt.cm.binary)
    plt.imshow(source_data[incorrect_image_index].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(errorTitle)
  plt.show()

def displayLossAndAccuracy(history):
  plt.plot(history.history['loss'], color='red', label='Loss')
  plt.plot(history.history['acc'], color='blue', label='Accuracy')
  plt.legend()
  plt.grid(b=True, which='major', color='#666666', linestyle='-')
  plt.show()

def printSampleIncorrectImages(image_data, image_labels, class_names, model):
    incorrects = np.nonzero(model.predict_classes(image_data).reshape((-1,)) != image_labels)
    
    plt.figure(figsize=(10,10))
    max_items = 25 if len(incorrects[0]) > 25 else len(incorrects[0])
    for i in range(max_items):
        incorrect_image_index = incorrects[0][i]
        incorrect_image_data = np.array([image_data[incorrect_image_index]])
        expected_class = class_names[image_labels[incorrect_image_index]]
        predicted_class = class_names[model.predict_classes(incorrect_image_data)[0]]
        errorTitle = "Expected class: '%s'\nPredicted class: '%s'"%(expected_class, predicted_class)
    
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



def displayLayerActivations(cnn_model, image):
  import numpy as np
  from tensorflow.keras.preprocessing.image import img_to_array, load_img

  # Let's define a new Model that will take an image as input, and will output
  # intermediate representations for all layers in the previous model after
  # the first.
  successive_outputs = [layer.output for layer in cnn_model.layers[1:]]
  #visualization_model = Model(img_input, successive_outputs)
  visualization_model = tf.keras.models.Model(inputs = cnn_model.input,
                                              outputs = successive_outputs)
  
  # Let's run our image through our network, thus obtaining all
  # intermediate representations for this image.
  image = image.reshape(1,28,28,1)
  successive_feature_maps = visualization_model.predict(image)

  # These are the names of the layers, so can have them as part of our plot
  layer_names = [layer.name for layer in model.layers]

  # Now let's display our representations
  for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
      # Just do this for the conv / maxpool layers, not the fully-connected layers
      n_features = feature_map.shape[-1]  # number of features in feature map
      # The feature map has shape (1, size, size, n_features)
      size = feature_map.shape[1]
      # We will tile our images in this matrix
      display_grid = np.zeros((size, size * n_features))
      for i in range(n_features):
        # Postprocess the feature to make it visually palatable
        image = feature_map[0, :, :, i]
        image -= image.mean()
        image /= image.std()
        image *= 64
        image += 128
        image = np.clip(image, 0, 255).astype('uint8')
        # We'll tile each filter into this big horizontal grid
        display_grid[:, i * size : (i + 1) * size] = image 
      # Display the grid
      scale = 20. / n_features
      plt.figure(figsize=(scale * n_features, scale))
      plt.title(layer_name)
      plt.grid(False)
      #plt.imshow(display_grid, aspect='auto', cmap='viridis')
      plt.imshow(display_grid, aspect='auto', cmap=plt.cm.binary)
      
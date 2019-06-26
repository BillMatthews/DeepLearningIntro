# Import the packages we need
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


def printLossAndAccuracy(history):
  import matplotlib.pyplot as plt
  history_dict = history.history

  acc = history_dict['acc']
  val_acc = history_dict['val_acc']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  
  plt.show()

  plt.clf()   # clear figure

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()


  plt.show()

def predictStarRating(review, model, word_index):
      encoded_review = [encodeReview(review, word_index)]
      sequence = keras.preprocessing.sequence.pad_sequences(encoded_review,
                                                      value=word_index["<PAD>"],
                                                      padding='post',
                                                      maxlen=256)
      prediction = model.predict(sequence)
      if prediction >= 0.9:
          return "5 Stars"
      if prediction >= 0.7:
          return "4 Stars"
      if prediction >= 0.5:
          return "3 Stars"
      if prediction >= 0.3:
              return "2 Stars"
      return "1 Star"
      

def getWordIndex(corpus):
  word_index = corpus.get_word_index()

  # The first indices are reserved
  word_index = {k:(v+3) for k,v in word_index.items()}
  word_index["<PAD>"] = 0
  word_index["<START>"] = 1
  word_index["<UNK>"] = 2  # unknown
  word_index["<UNUSED>"] = 3

  reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  

  return word_index, reverse_word_index

def decodeReview(text, reverse_word_index):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def encodeReview(text, word_index):
  remove_words = ["!", "$", "%", "&", "*", ".", "?", "<", ">", ","]  
  for word in remove_words:
      text = text.replace(word, "")
  
  return [word_index[token] if token in word_index else 2 for token in text.lower().split(" ")]

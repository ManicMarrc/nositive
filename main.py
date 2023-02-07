import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

import argparse
import random

def load_data(data_path: str, max: int = 50_000) -> tf.data.Dataset:
  texts = []
  labels = []
  with open(data_path, 'r') as f:
    lines = f.readlines()
    for _ in range(max):
      line = random.choice(lines)
      values = line[1:-2].split('","')
      texts.append(values[-1])
      labels.append(int(values[0]) // 2)
  dataset = tf.data.Dataset.from_tensor_slices((texts, keras.utils.to_categorical(labels, 3))).batch(32)

  return dataset

def standardize(s: tf.string) -> tf.string:
  s = tf.strings.lower(s)
  s = tf.strings.regex_replace(s, '@[a-zA-Z0-9_]* ', '')
  return s

def init_vectorizer(example_dataset: tf.data.Dataset) -> layers.TextVectorization:
  vectorizer = layers.TextVectorization(standardize=standardize, max_tokens=20_000, output_mode='int', output_sequence_length=500)
  vectorizer.adapt(example_dataset)
  return vectorizer

def prepare_data(dataset: tf.data.Dataset, vectorizer: layers.TextVectorization) -> tf.data.Dataset:
  return dataset.map(lambda x, y: (vectorizer(tf.expand_dims(x, axis=-1)), y)).cache().prefetch(10)

def get_model(train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, vectorizer: layers.TextVectorization) -> keras.Model:
  try:
    with keras.utils.custom_object_scope({'standardize': standardize}):
      return keras.models.load_model('model.tf')
  except (ImportError, IOError):
    inputs = keras.Input(shape=(None, ), dtype='int64')
    x = layers.Embedding(20_000, 128)(inputs)
    x = layers.Conv1D(128, kernel_size=4, padding='valid', activation='relu', strides=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(64, kernel_size=5, padding='valid', activation='relu', strides=2)(x)
    x = layers.Conv1D(64, kernel_size=6, padding='valid', activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(3, activation='softmax')(x)
    model = keras.Model(inputs, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_dataset, batch_size=32, epochs=50)
    model.evaluate(train_dataset)

    inputs = keras.Input(shape=(1,), dtype='string')
    x = vectorizer(inputs)
    x = model(x)
    e2e_model = keras.Model(inputs, x)
    e2e_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    e2e_model.save('model.tf')
    return e2e_model

def main():
  parser = argparse.ArgumentParser(prog='nesitive', description='Returns whether the text that has been inputed is negative or positive.')
  parser.add_argument('input')
  args = parser.parse_args()

  train_dataset = load_data('datasets/train_data.csv')
  test_dataset = load_data('datasets/test_data.csv')

  vectorizer = init_vectorizer(train_dataset.map(lambda x, y: x))

  train_dataset = prepare_data(train_dataset, vectorizer)
  test_dataset = prepare_data(test_dataset, vectorizer)

  model = get_model(train_dataset, test_dataset, vectorizer)
  result = model(np.array([args.input])).numpy()
  match result.argmax():
    case 0: print('The text is negative!')
    case 1: print('The text is neutral!')
    case 2: print('The text is positive!')

if __name__ == '__main__':
  main()

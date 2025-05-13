
# Approach uses integer encoding + recurrent neural networks

import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

train_file_path = "data/train-data.tsv"
test_file_path = "data/valid-data.tsv"

# create dataframe for data
train_data = pd.read_csv(test_file_path, sep="\t", names=["class", "text"])

test_data = pd.read_csv(test_file_path, sep="\t", names=["class", "text"])

# convert categorical data into numerical data (class)
train_df1 = pd.get_dummies(train_data['class'], dtype='int')
test_df1 = pd.get_dummies(test_data['class'], dtype='int')
train_data = pd.concat([train_data, train_df1], axis=1).reindex(train_data.index)
train_data.drop('class', axis=1, inplace=True)
test_data = pd.concat([test_data, test_df1], axis=1).reindex(test_data.index)
test_data.drop('class', axis=1, inplace=True)
print(train_data.loc[0, 'text'])
print(train_data.head())
# Encode text
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_data['text'])
vocab = np.array(encoder.get_vocabulary())


# create the model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
print(train_data['spam'])

train_labels = train_data['spam'].astype('float32').values
test_labels = test_data['spam'].astype('float32').values
train_texts = train_data['text'].astype(str).values
test_texts = test_data['text'].astype(str).values
print(train_texts)
print(train_labels)
history = model.fit(train_texts, train_labels, epochs=20,
                    validation_data=(test_texts, test_labels))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
  prediction = model.predict(tf.convert_to_tensor([pred_text]))
  result = [prediction[0][0], 'ham' if prediction[0][0] < 0.5 else 'spam']

  return (result)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()

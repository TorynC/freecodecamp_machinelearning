
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass'''

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Normalization, StringLookup, CategoryEncoding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data/insurance.csv')

# convert categorical datatypes to numerical
# dummy for sex
df1 = pd.get_dummies(dataset['sex'], dtype='int')
# dummy for smoker
df2 = pd.get_dummies(dataset['smoker'], dtype='int')
# dummy for region
df3 = pd.get_dummies(dataset['region'], dtype='int')

# concatonate all dummy data frames into original dataframe
dataset = pd.concat([dataset, df1, df2, df3], axis=1).reindex(dataset.index)
dataset.drop('sex', axis=1, inplace=True)
dataset.drop('smoker', axis=1, inplace=True)
dataset.drop('region', axis=1, inplace=True)

# split data into training and testing data
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, train_size=0.8,
                                               random_state=0)
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# dataframe now has 11 features used to predict expenses

# Creating the model
model = Sequential()
model.add(Dense(128, input_shape=(11,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['mae', 'mse'])

# training the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(train_dataset, train_labels, epochs=50, validation_split=0.2, callbacks=[early_stop])

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)

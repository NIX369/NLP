import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

dataset, info = tfds.load('glue/sst2', with_info = True)

print(info.features)
print(info.features['label'].num_classes)
print(info.features['label'].names)

dataset_train, dataset_validation = dataset['train'], dataset['validation']
dataset_train

for example in dataset_train.take(2):
  review, label = example['sentence'], example['label']
  print(review)
  print(label.numpy())

training_reviews = []
training_labels = []

validation_reviews = []
validation_labels = []

for item in dataset_train.take(10000):
  review, label = item['sentence'], item['label']
  training_reviews.append(str(review.numpy()))
  training_labels.append(label.numpy())

print("Length of training Reviews is : ", len(training_reviews))

for i in range(0, 2):
  print(training_reviews[i])
  print(training_labels[i])

for item in dataset_validation.take(-1):
  review, label = item['sentence'], item['label']
  validation_reviews.append(str(review.numpy()))
  validation_labels.append(label.numpy())

print("Length of validation reviews is : ", len(validation_reviews))

for i in range(0, 2):
  print(validation_reviews[i])
  print(validation_labels[i])

# Tokenizing the words

vocab_size = 4000
embedding_dim = 16
max_length = 50
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_token)
tokenizer.fit_on_texts(training_reviews)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_reviews)
training_padded = pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_reviews)
validation_padded = pad_sequences(validation_sequences, maxlen = max_length, padding=padding_type, truncating=trunc_type)

training_labels_final = np.array(training_labels)
validation_labels_final = np.array(validation_labels)

model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
      tf.keras.layers.GlobalAveragePooling1D(),
      tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(training_padded, training_labels_final, epochs = num_epochs, validation_data=(validation_padded, validation_labels_final))

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

review1 = """I loved this movie"""

review2 = """that was the worst movie I've ever seen"""

review3 = """too much violence even for a Bond film"""

review4 = """a captivating recounting of a cherished myth"""

new_reviews = [review1, review2, review3, review4]

# Define a function to prepare the new reviews for use with a model
# and then use the model to predict the sentiment of the new reviews           

def predict_review(model, reviews):
  # Create the sequences
  padding_type='post'
  sample_sequences = tokenizer.texts_to_sequences(reviews)
  reviews_padded = pad_sequences(sample_sequences, padding=padding_type, 
                                 maxlen=max_length) 
  classes = model.predict(reviews_padded)
  for x in range(len(reviews_padded)):
    print(reviews[x])
    print(classes[x])
    print('\n')

predict_review(model, new_reviews)

"""## Define a function to train and show the results of models with different layers"""

def fit_model_and_show_results(model, reviews):
  model.summary()
  history = model.fit(training_padded, training_labels_final, epochs=num_epochs, validation_data=(validation_padded, validation_labels_final))

  plot_graphs(history, "accuracy")
  plot_graphs(history, "loss")
  predict_review(model, reviews)

"""## USING A CNN"""

num_epochs = 30

model_cnn = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
      tf.keras.layers.Conv1D(16, 5, activation='relu'),
      tf.keras.layers.GlobalMaxPool1D(),
      tf.keras.layers.Dense(1, activation = 'sigmoid')
])

lr = 0.0001
model_cnn.compile(loss = 'binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr), metrics = ['accuracy'])

fit_model_and_show_results(model_cnn, new_reviews)

"""## USING A GRU"""

num_epochs = 30

model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.00003 # slower than the default learning rate
model_gru.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])

fit_model_and_show_results(model_gru, new_reviews)

"""## USING A BIDIRECTIONAL LSTM"""

num_epochs = 30

model_bidi_lstm = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
      tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.00003
model_bidi_lstm.compile(loss='binary_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(learning_rate),
                        metrics=['accuracy'])
fit_model_and_show_results(model_bidi_lstm, new_reviews)

"""## USING MULTIPLE BIDIRECTIONAL LSTMs"""

num_epochs = 30

model_multiple_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.0003
model_multiple_bidi_lstm.compile(loss='binary_crossentropy',
                                 optimizer=tf.keras.optimizers.Adam(learning_rate),
                                 metrics=['accuracy'])
fit_model_and_show_results(model_multiple_bidi_lstm, new_reviews)

# Write some new reviews 

review1 = """I loved this movie"""

review2 = """that was the worst movie I've ever seen"""

review3 = """too much violence even for a Bond film"""

review4 = """a captivating recounting of a cherished myth"""

review5 = """I saw this movie yesterday and I was feeling low to start with,
 but it was such a wonderful movie that it lifted my spirits and brightened 
 my day, you can\'t go wrong with a movie with Whoopi Goldberg in it."""

review6 = """I don\'t understand why it received an oscar recommendation
 for best movie, it was long and boring"""

review7 = """the scenery was magnificent, the CGI of the dogs was so realistic I
 thought they were played by real dogs even though they talked!"""

review8 = """The ending was so sad and yet so uplifting at the same time. 
 I'm looking for an excuse to see it again"""

review9 = """I had expected so much more from a movie made by the director 
 who made my most favorite movie ever, I was very disappointed in the tedious 
 story"""

review10 = "I wish I could watch this movie every day for the rest of my life"

more_reviews = [review1, review2, review3, review4, review5, review6, review7, 
               review8, review9, review10]

print("============================\n","Embeddings only:\n", "============================")
predict_review(model, more_reviews)

print("============================\n","With CNN\n", "============================")
predict_review(model_cnn, more_reviews)

print("===========================\n","With bidirectional GRU\n", "============================")
predict_review(model_gru, more_reviews)

print("===========================\n", "With a single bidirectional LSTM:\n", "===========================")
predict_review(model_bidi_lstm, more_reviews)

print("===========================\n", "With multiple bidirectional LSTM:\n", "==========================")
predict_review(model_multiple_bidi_lstm, more_reviews)


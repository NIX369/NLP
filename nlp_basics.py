import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'What is up with nikhil?',
    'Google form'
]

# WORD BASED ENCODINGS - TOKENIZING
# OOV - Out Of Vocabulary token
tokenizer = Tokenizer(oov_token = "OOV_TOKEN")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
sequences = pad_sequences(sequences)
print(sequences)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
      'I love my dog',
      'I love my cat',
      'You love my dog',
      'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)
print("\n Word Index : ", word_index)
print("\n Sequences : ", sequences)
print("\n Padded sequence : ")
print(padded)

test_data = [
      'I really love my dog',
      'my dog loves my manatee'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
padded_test_sequences = pad_sequences(test_sequences)

print("\n Test Sequence : ", test_sequences)
print("\n Padded Test Sequnce : ")
print(padded_test_sequences)


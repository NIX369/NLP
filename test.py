# # TOKENIZATION
# from keras.preprocessing.text import Tokenizer
#
# sentences = ["I love my cat",
#              "I love my dog",
#              "I hate your dog!"]
#
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(sentences)
# word_index = tokenizer.word_index
# print(word_index)


# TURNING SENTENCECS INTO DATA

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = ["I love my cat",
             "I love my dog",
             "I hate your dog!",
             "I think your dog looks amazing!"]

tokenizer = Tokenizer(oov_token="<00V>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding="post", truncating="pre", maxlen= 5)  # padding is done to make all the sentences to be of same length (padding is done to the sequence)
                                                                                # padding =  "post": this argument puts all the zeros to the end of sentence
print(word_index)
print(sequences)
print(padded)


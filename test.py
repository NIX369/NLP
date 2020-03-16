# TOKENIZATION
from keras.preprocessing.text import Tokenizer

sentences = ["I love my cat",
             "I love my dog",
             "I hate your dog!"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)




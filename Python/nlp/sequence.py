from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "I love my dog",
    "I love my cat",
    " Do you love  my  cat?"
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

print("word_index:", word_index)
print("sequences:", sequences)

test_sentences = [
    "I really love my dog"
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
print("test_sequences:", test_sequences)
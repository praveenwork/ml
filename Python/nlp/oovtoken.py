from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "I Love my dog",
    "I love my cat",
    "Do you love my dog?"
]


tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

print("word index:", word_index)
print("sequences :", sequences)

test_sentences = [
    "Do you really love my dog?",
    "Do you want my dog?"
]

test_sequences =  tokenizer.texts_to_sequences(test_sentences)

print("test_sequences:", test_sequences)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love my dog",
    "I love my cat",
    "you love  my dog!",
    "Do  you think my dog is amazing"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences  = tokenizer.texts_to_sequences(sentences)

padding = pad_sequences(sequences, padding="post", truncating="post", maxlen=6)

print("Word Index: ", word_index)
print("Sequences: ", sequences)
print("Padding: ", padding)
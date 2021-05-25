from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "I Love My Dog",
    "i love my cat",
    "You love my DOG!"
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(word_index)
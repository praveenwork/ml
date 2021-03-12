import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np
import matplotlib.pyplot as plt

# Read json and the data
with open("sarcasm.json", "r") as f:
    dataset = json.load(f)


sentences = []
labels = []

for item in dataset:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

#Split data as trainging and testing

training_size = 20000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


#Hyper Perameters

VOCAB_SIZE = 5000
OOV_TOKEN = "<OOV>"
MAX_LEN = 90
TRUNCATING = 'post'
EMBEDDED_DIM = 16
PADDING_TYPE = 'post'

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

reverse_index = [(value, key) for (key, value) in word_index.items()]

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences=sequences, maxlen=MAX_LEN, padding=PADDING_TYPE, truncating=TRUNCATING)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded = pad_sequences(test_sequences, maxlen=MAX_LEN, padding= PADDING_TYPE, truncating=TRUNCATING)

print(padded.shape)

training_padded_final = np.array(padded)
training_labels_final = np.array(training_labels)

training_padded_final = np.array(padded)
training_labels_final = np.array(training_labels)
validation_padded_final = np.array(test_padded)
validation_labels_final = np.array(testing_labels)


#define model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDED_DIM, input_length=MAX_LEN),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer="adam", loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit(
    training_padded_final,
    training_labels_final,
    epochs=20,
    validation_data=(validation_padded_final,validation_labels_final),
    verbose=1
)

accuracy = history.history['accuracy']
loss = history.history['loss']

validation_accuracy = history.history['val_accuracy']
validation_loss = history.history['val_loss']

plt.plot(accuracy)
plt.plot(validation_accuracy)

plt.figure()


plt.plot(loss)
plt.plot(validation_loss)

plt.show()



import tensorflow  as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for data in datastore:
    sentences.append(data['headline'])
    labels.append(data['is_sarcastic'])

VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
EMBEDDED_DIM = 16
MAX_LEN = 50
TRUNCATING = 'post'
TRAINING_DATA_SIZE = 20000

training_sentences = sentences[0:TRAINING_DATA_SIZE]
testing_sentences = sentences[TRAINING_DATA_SIZE:]

training_labels = labels[0:TRAINING_DATA_SIZE]
testing_labels = labels[TRAINING_DATA_SIZE:]

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padding = pad_sequences(sequences, maxlen=MAX_LEN, truncating=TRUNCATING)

test_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padding = pad_sequences(test_sequences, maxlen=MAX_LEN, truncating=TRUNCATING)


testing_labels = np.array(testing_labels)
test_padding = np.array(test_padding)
padding = np.array(padding)
training_labels = np.array(training_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDED_DIM, input_length=MAX_LEN),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(24, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()
history = model.fit(
    padding,
    training_labels,
    epochs=30,
    validation_data=(test_padding, testing_labels),
    verbose=1
)

def plot_graphs(history, string):
    accuracy = history.history[string]
    val_accuracy = history.history['val_' + string]
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.figure()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
plt.show()




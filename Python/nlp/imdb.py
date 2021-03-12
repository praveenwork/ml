import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import io

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True )

training_data, testing_data = imdb['train'], imdb['test']

training_sentences = []
testing_sentence = []

testing_labels = []
training_labels = []

print(len(training_data))
print(len(testing_data))

for s, l in training_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy())

for s, l in testing_data:
    testing_sentence.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

testing_labels_final = np.array(testing_labels)
training_labels_final = np.array(training_labels)


VOCAB_SIZE = 10000
OOV_TOKEN = "<OOV>"
MAX_LEN = 120
TRUNCATE = 'post'
EMBEDDED_DIM = 16

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

reverse_word_index = dict((value, key) for key, value in word_index.items())

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences=sequences, maxlen=MAX_LEN, truncating=TRUNCATE)

test_sequences = tokenizer.texts_to_sequences(testing_sentence)
test_padded = pad_sequences(sequences=test_sequences, maxlen=MAX_LEN, truncating=TRUNCATE)

# model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDED_DIM, input_length=MAX_LEN),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

history = model.fit(
    padded,
    training_labels_final,
    epochs=10,
    validation_data=(test_padded,testing_labels_final),
    verbose=1
)

model.summary()

train_accuracy = history.history['accuracy']
train_loss = history.history['loss']

val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(train_accuracy))
plt.plot(epochs, train_accuracy)
plt.plot(epochs, val_accuracy)
plt.figure()

plt.plot(epochs, train_loss)
plt.plot(epochs, val_loss)
plt.figure()
plt.show()

#weights
e = model.layers[0]
weights = e.get_weights()[0]

# generate files for projector tensorflow

out_v = io.open("vec.tsv", "w", encoding='utf-8')
out_m = io.open("met.tsv", "w", encoding="utf-8")

for word_num in range(1, VOCAB_SIZE):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

print("Done")







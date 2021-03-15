import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import io

#Download the dataset
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
print(imdb)
train, test, unsupervised = imdb['train'], imdb['test'], imdb['unsupervised']
tokenizer = info.features['text'].encoder
print(tokenizer.subwords)

sample_string = "Tensorflow, from basic to to a mastery"

tokenized_string = tokenizer.encode(sample_string)
print("Tokenized String :{}".format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print("Original String :{}".format(original_string))


for ts in tokenized_string:
    print("{}----> {} ".format(ts, tokenizer.decode([ts])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test))

embedded_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedded_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics= ['accuracy'])
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset
)

accuracy = history.history['accuracy']
loss = history.history['loss']

val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

def plotGraphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel('string')
    plt.legend(string, 'val_'+string)
    plt.show()
#plotGraphs(history,'accuracy')
#plotGraphs(history,'loss')


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
out_v = io.open("subtoken.tsv", 'w', encoding="utf-8")
out_m = io.open("subtoken_m.tsv", "w", encoding="utf-8")
for word_num in range(1,tokenizer.vocab_size):
    word = tokenizer.decode([word_num])
    embedding = weights[word_num]
    out_m.write(word+"\n")
    out_v.write("\t".join([str(x) for x in embedding]) + "\n")
out_v.close()
out_m.close()



import pathlib
import tensorflow as tf
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file('flower_photos', origin=url, untar=True,md5_hash=None )
data_dir = pathlib.Path(data_dir)

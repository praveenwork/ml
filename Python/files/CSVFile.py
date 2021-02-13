import os
import csv
import numpy as np
def file_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file,  delimiter = ',')
        first_row = True
        temp_labels = []
        temp_images = []
        for row in csv_reader:
            if first_row:
                first_row = False
            else:
                temp_labels.append(row[0])
                image = row[1:785]
                image_to_array = np.array_split(image, 28)
                temp_images.append(image_to_array)

        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')

        return images, labels

local_file = "/Users/ppasupuleti/Praveen/Project/ML/Data/HandSign/sign_mnist_train.csv"
images, labels = file_data(local_file)
print(labels.shape)
print(images.shape)
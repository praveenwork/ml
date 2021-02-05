import os
import zipfile

zipFilePath = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/PerImages.zip"
sourceData = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/PerImages"
dataPath = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/"

# Extract the zip
zip_ref = zipfile.ZipFile(zipFilePath)
zip_ref.extractall(sourceData)
zip_ref.close()

# Training
trainingSourcePath = os.path.join(dataPath, "Training")
trainingCatsPath = os.path.join(trainingSourcePath, "Cats")
trainingDogsPath = os.path.join(trainingSourcePath, "Dogs")
os.mkdir(trainingSourcePath)
os.mkdir(trainingCatsPath)
os.mkdir(trainingDogsPath)

# Testing
testingSourcePath = os.path.join(dataPath, "Testing")
testingCatPath = os.path.join(testingSourcePath, "Cats")
testingDogPath = os.path.join(testingSourcePath, "Dogs")
os.mkdir(testingSourcePath)
os.mkdir(testingCatPath)
os.mkdir(testingDogPath)



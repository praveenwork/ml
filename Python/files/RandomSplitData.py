import os
import random
import zipfile
import shutil

def split_data(source, training, testing, split):

    # check size of the source folder
    source_files = os.listdir(source)
    files_count = len(source_files)
    if files_count > 0:
        # Shuffle the files
        random.shuffle(source_files)
        # split the files
        target_list = source_files[:int(files_count * split)]
        testing_list = source_files[:files_count-len(target_list)]
        print("targetList count", len(target_list))
        print("testingList Count", len(testing_list))

        # copy files from source to destinations
        for file in target_list:
            if os.path.getsize(os.path.join(source, file)) > 0:
                shutil.copyfile(os.path.join(source, file), os.path.join(training, file))

        for file in testing_list:
            if os.path.getsize(os.path.join(source, file)) > 0:
                shutil.copyfile(os.path.join(source, file), os.path.join(testing, file))
    else:
        print("Source {0} Directory is Empty", source)


zipFilePath = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/PerImages.zip"
sourceData = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/PerImages"
dataPath = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/"

# Check folder exist or not

if os.path.exists(sourceData):
    shutil.rmtree(sourceData)
    shutil.rmtree(os.path.join(dataPath, "Training"))
    shutil.rmtree(os.path.join(dataPath, "Testing"))


# Extract the zip
zip_ref = zipfile.ZipFile(zipFilePath)
zip_ref.extractall(dataPath)
zip_ref.close()

# Training
trainingSourcePath = os.path.join(dataPath, "Training")
trainingCatsPath = os.path.join(trainingSourcePath, "Cats")
trainingDogsPath = os.path.join(trainingSourcePath, "Dogs")
#os.mkdir(trainingSourcePath)
#os.mkdir(trainingCatsPath)
#os.mkdir(trainingDogsPath)
os.makedirs(trainingCatsPath)
os.makedirs(trainingDogsPath)
# Testing
testingSourcePath = os.path.join(dataPath, "Testing")
testingCatPath = os.path.join(testingSourcePath, "Cats")
testingDogPath = os.path.join(testingSourcePath, "Dogs")
os.mkdir(testingSourcePath)
os.mkdir(testingCatPath)
os.mkdir(testingDogPath)


splitPercentage = .9
# Split Cats Data
split_data(os.path.join(sourceData, "Cats"), trainingCatsPath, testingCatPath, splitPercentage )

# Split Dogs Data
split_data(os.path.join(sourceData, "Dogs"), trainingDogsPath, testingDogPath, splitPercentage )









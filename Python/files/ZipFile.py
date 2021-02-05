import zipfile

file = "/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/PerImages.zip"

zip_ref = zipfile.ZipFile(file)
zip_ref.extractall("/Users/ppasupuleti/Praveen/Project/ML/Data/CatsNDogs/")
zip_ref.close()

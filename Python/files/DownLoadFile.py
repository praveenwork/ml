import requests

def download_file(url, filepath):
    req = requests.get(url)
    open(filepath, "wb").write(req.content)


# sarcasm data
#download_file("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json", "../nlp/sarcasm1.json")

# BBC DataSet
download_file("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv"    , "../nlp/bbc-text.csv")

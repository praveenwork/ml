import requests

url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
req = requests.get(url)

open("sarcasm.json", "wb").write(req.content)


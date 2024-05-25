import os
import zipfile

local_zip = 'C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\Natural Language Processing\\archive (1).zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\Natural Language Processing')
zip_ref.close()

import json

json_file_path = 'C:\\Users\\TANMAY KAMEWAL\\OneDrive\\Desktop\\DeepLearning\\Natural Language Processing\\Sarcasm_Headlines_Dataset_v2.json'

sentences = []
labels = []
urls = []

with open(json_file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])
        urls.append(data['article_link'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))
# print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)





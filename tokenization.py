from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "My Name is Tanmay Kamewal",
    "I am a student",
    "I am a teacher",
    "I am a programmer",
    "good Byy"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
print(len(word_index))
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
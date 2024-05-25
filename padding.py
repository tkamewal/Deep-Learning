from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
    ]
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
print("\nWord Index = " , word_index)
print("\nSequences = ", sequences)
print("\nPadded Sequences:")
print(padded)
print("\nPadded Sequences with post padding:")
padded = pad_sequences(sequences, padding='post')
print(padded)
print("\nPadded Sequences with pre padding:")
padded = pad_sequences(sequences, padding='pre')
print(padded)
print("\nPadded Sequences with truncating:")
padded = pad_sequences(sequences, padding='post', maxlen=5)
print(padded)

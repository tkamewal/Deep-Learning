from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

import tensorflow as tf

tokenizer = Tokenizer()
data = """Sun zaalima mere sanu koi darr na\n
Ki samjhega zamana\n
Oh tu vi si kamli\n
Main vi sa kamla\n
Ishqe da rog seyana\n
Ishqe da rog seyana\n
Sun mere humsafar\n
Kya tujhe itni si bhi khabar\n
Sun mere humsafar\n
Kya tujhe itni si bhi khabar\n
Ki teri saanse chalti jidhar\n
Rahunga bas wahi umrr bhar\n
Rahunga bas wahi umrr bhar haaye\n
Jitni haseen ye mulakatein hai\n
Unse bhi pyari teri baatein hai\n
Baaton mein teri jo kho jaate hai\n
Aaun na hosh mein main kabhi\n
Baahon mein hai teri zindagi haaye\n
Sun mere humsafar\n
Kya tujhe itni si bhi khabar\n
Zaalima tere ishq ch main\n
Ho gayi aan kamli haye\n
Main toh yoon khada kis\n
Soch mein pada tha\n
Kaise jee raha tha main deewana\n
Chupke se aake tune\n
Dil mein sama ke tune\n
Chhed diya kaisa ye fasana\n
Oh muskurana bhi tujhi se sikha hai\n
Dil lagane ka tu hi tareeka hai\n
Aitbaar bhi tujhi se hota hai\n
Aaun na hosh mein main kabhi\n
Bahon mein hai teri zindagi haaye\n
Hai nahi tha pata\n
Ke tujhe maan lunga khuda\n
Ki teri galliyon mein iss kadar\n
Aaunga har paher\n
Sun mere humsafar\n
Kya tujhe itni si bhi khabar\n
Ki teri saanse chalti jidhar\n
Rahunga bas wahi umrr bhar\n
Rahunga bas wahi umrr bhar haaye\n
(Zaalima tere ishq ch main)\n
Hun Kehndi Sorry Lyrics Mavi Singh\n"""

corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(total_words)
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
print(xs[5])
print(ys[5])
print(tokenizer.word_index)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

history = model.fit(xs, ys, epochs=300, verbose=1, callbacks=[callback()])

import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()

#predicting Music
seed_text = "Sun zaalima mere sanu koi darr na"
next_words = 100
for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    import numpy as np

    predicted = model.predict(token_list, verbose=0)
    predicted_classes = np.argmax(predicted, axis=-1)

    output_word = ""
    for word, index in tokenizer.word_index.items():
        if any(index == pred for pred in predicted.flatten()):
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.document_count)
print(tokenizer.word_docs)
print(tokenizer.index_docs)








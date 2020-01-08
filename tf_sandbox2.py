## REQUIRES NUMPY 1.16.1
## TO USE DO: pip install numpy==1.16.1

import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

## gets index word tuples
word_index = data.get_word_index()

## breaks up tuples to give in dictionary form
## add 3 to values for special words below
word_index = {k:(v+3) for k,v in word_index.items()}
## PAD: makes all reviews same length
## START: start of review
## UNK: unknown word stand-in
## UNUSED: for any unused words?
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

## reverses dictionary so INT -> WORD
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])


## Pads and cuts all data to have length 250
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value= word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value= word_index["<PAD>"], padding="post", maxlen=250)



def decode_review(text):
    return "  ".join([reverse_word_index.get(i,"?") for i in text])



## MODEL
model = keras.Sequential()
#word embedding layer, 16D embedding vector
model.add(keras.layers.Embedding(10000,16))
#averages word embeddings
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

## Splits train and validation data sets
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]



fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=2)

results = model.evaluate(test_data, test_labels)

model.save("movie_review_model.h5")

'''
model= keras.models.load_model("movie_review_model.h5")

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)
    return encoded

with open("parasite_review.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value= word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])





print("model results: ", results)

test_review= test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction:"+str(predict[0]) )
print("Actual: "+str(test_labels[0]))
'''





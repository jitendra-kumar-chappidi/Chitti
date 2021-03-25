"""
predict the sentiment of movie reviews

input data: IMDB dataset
libraries used: keras, tesorflow, matplotlib, numpy
Author : Jitendra kumar chappidi

"""


# import
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from matplotlib import pyplot
import warnings

# ignoring warnings
warnings.filterwarnings("ignore")

top_words = 5000  # number of top words from input data
review_word_length = 500  # number of words in each review
epoch = 2  # number of iterations through entire dataset
batch_size = 128  # batch of input data fed to model


# # load dataset
# (X_train, y_train), (X_test, y_test) = imdb.load_data()

# load dataset but only keep top_words , set rest to zero
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# padding the reviews to maximum words length of "review_word_length"
X_train = sequence.pad_sequences(X_train, maxlen=review_word_length)
X_test = sequence.pad_sequences(X_test, maxlen=review_word_length)

# # concatenate data
# X = np.concatenate((X_train, X_test))
# y = np.concatenate((y_train, y_test))


# # summary of input dataset ( understanding data )
# print("Training Data Summary: ")
# print("Input data shape: ", X.shape)
# print("Output data shape: ", y.shape)
#
# # summary of  classes in output data
# print("Summary of number of classes in output data: ", np.unique(y))
#
# # number of unique words in input data
# print(f"Number of unique words in input data: {len(np.unique(np.hstack(X)))}")
#
# # summary of reviews length
# review_len = [len(x) for x in X]
# print(f"average review length: {np.mean(review_len)}, number of words: {np.std(review_len)}")
#
# # plot the the reviews
# pyplot.boxplot(review_len)
# pyplot.show()

#
# # model single layer perception
# model = Sequential()
# model.add(Embedding(top_words, 32, input_length=review_word_length))
# model.add(Flatten())
# model.add(Dense(250, activation="relu"))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# 1D CNN model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=review_word_length))
model.add(Conv1D(kernel_size=3, activation="relu", filters=32, padding="same"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# summary of the model
model.summary()


# model training / fitting model to input data
print(f"Training on {len(X_train)} , Validation on {len(X_test)}")
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epoch, verbose=2)

# evaluation of model
predictions_score = model.evaluate(X_test, y_test, verbose=0)

# results
print(f"Model accuracy: {predictions_score[1]*100}")


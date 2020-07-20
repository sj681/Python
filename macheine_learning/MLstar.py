import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt


#loads in the np arrays which contain the images (spectra)
#There are four total classes GD, DS, Hybrid and UNCLASSIFIED

train_Xgd1 = np.load('New_GdArray.npy')
train_Xds1 = np.load('New_DsArray.npy')
train_Xhyb1 = np.load('New_HybridArray.npy')
train_Xnon1 = np.load('New_UnArray.npy')

train_Xgd =train_Xgd1.transpose(2, 0, 1)
train_Xds =train_Xds1.transpose(2, 0, 1)
train_Xhyb =train_Xhyb1.transpose(2, 0, 1)
train_Xnon =train_Xnon1.transpose(2, 0, 1)

#Gives each star type a number.
#GD: 1
#DS: 2
#HYB: 3
#UN: 0
train_Ygd=np.full(train_Xgd.shape[0], 1)
train_Yds=np.full(train_Xds.shape[0], 2)
train_Yhyb=np.full(train_Xhyb.shape[0], 3)
train_Ynon=np.full(train_Xnon.shape[0], 0)

# Loading the training as the first 1000 images and
# uses the last 259 as test images
#splits the data into training and test data.

test_Xgd = train_Xgd[1000::]
test_Ygd = train_Ygd[1000::]
train_Xgd = train_Xgd[:1000:]
train_Ygd = train_Ygd[:1000:]

test_Xds = train_Xds[1000::]
test_Yds = train_Yds[1000::]
train_Xds = train_Xds[:1000:]
train_Yds = train_Yds[:1000:]

test_Xhyb = train_Xhyb[1000::]
test_Yhyb = train_Yhyb[1000::]
train_Xhyb = train_Xhyb[:1000:]
train_Yhyb = train_Yhyb[:1000:]

test_Xnon = train_Xnon[1000::]
test_Ynon = train_Ynon[1000::]
train_Xnon = train_Xnon[:1000:]
train_Ynon = train_Ynon[:1000:]

#concatenates all of the test / train data for the 4 star types into one array.
test_X = np.concatenate([test_Xnon, test_Xgd, test_Xds, test_Xhyb])
test_Y = np.concatenate([test_Ynon, test_Ygd, test_Yds, test_Yhyb])
train_X = np.concatenate([train_Xnon, train_Xgd, train_Xds, train_Xhyb])
train_Y = np.concatenate([train_Ynon, train_Ygd, train_Yds, train_Yhyb])

#randomly mixes up the training data so its not in order of the four categories but mixed.
permutation = np.random.permutation(len(train_X))
train_X = train_X[permutation]
train_Y = train_Y[permutation]

print("Train Permutation: ",permutation)

# Keeping the final permutation saved for safety.
text_file = open("FinalTrainPermutation.txt", "w")

# Reshapes the array into one suitable for CNN
train_X = train_X.reshape(-1, 480, 640, 1)
test_X = test_X.reshape(-1, 480, 640, 1)
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
# Takes it down to 0 or 1 pixel values. To simplify.
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding meaning the model cannot say that class 3 is more important
# than class 2.
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Splits training data into 80% training 20% validation
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.3, random_state=13)

# Number of images per batch, updated weights afterwards. Small is faster and easier, but less accurate at
# guessing the gradient, as high number batches have a lot more info per update.
batch_size = 2
# Number of runs through all data.
epochs = 20
num_classes = 4




# Every layer of the algorithm added one at a time. Sequential model with the input shape we used earlier. Linear
# activation with LeakyRelu added in the layer after.
classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(480, 640, 1), padding='same'))
classifier.add(LeakyReLU(alpha=0.1))


classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Conv2D(64, (2, 2), activation='linear', padding='same'))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
classifier.add(Flatten())
# Prevents overfitting
classifier.add(Dropout(0.5))
classifier.add(Dense(64, activation='linear'))
classifier.add(LeakyReLU(alpha=0.1))
# 4: class 0 or class 1 2 3
classifier.add(Dense(4, activation='softmax'))

# Finally compiling all layers with the most accurate lr and metric. Categorical gives us prediction weightings too, I
# think. Still, can't use binary so.
classifier.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.SGD(lr=0.00001, momentum=0, decay=0.0, nesterov=False),
                   metrics=['accuracy'])

# Shows some parameters in each layer and the total parameters
print(classifier.summary())

# Trains the model on the train_X data with their labels. We see variables we set earlier, validation is checking itself
# with each epoch. Verbose says how much info it gives you as it runs. It will take a while, don't think it's frozen.
history = classifier.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                               validation_data=(valid_X, valid_label))




# Sets The predicted classes as what we predicted
predicted_classes = classifier.predict(test_X)


# Saving the predictions and weightings for use in star analysis. before softmax. We do the same when predicting on
# all KASOC data with the same format. So simple, but so vital. This is how we find results and eliminate low confidence.
predic = 0
text_file = open("Predictions.txt", "w")

for predictionite in predicted_classes:

    text_file.write(str(predic) + ":" + str(predictionite)+"\n")
    predic += 1
text_file.close()


# Takes the max value and its position(?) in the array to find the true class. [0,1...0] -> class 1. Basically
# reverses one hot encoding.
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
print("c ", predicted_classes)
text_file = open("Predictions_max.txt", "w")

for i in range(len(predicted_classes)):
    text_file.write(str(i) + ":" + str(predicted_classes[i]) +"\t" +str(test_Y[i])+"\n")

text_file.close()


# Decide what is correct here by comparing to our initial labels.
correct = np.where(predicted_classes == test_Y)[0]
print("Final result 20 epochs Found %d correct labels" % len(correct))

# More details on accuracy. Prints out our total correct predictions in case of crashing I guess.
from sklearn.metrics import classification_report


target_names = ["Class {}".format(i) for i in range(num_classes)]
print((classification_report(test_Y, predicted_classes, target_names=target_names)))

text_file = open("Output.txt", "w")
text_file.write("\n Final result 20 epochs Found %d correct labels" % len(correct))
text_file.close()


# This plots the accuracy over time. Was good for getting a cool graph for the thesis.



# Same for incorrect
incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))


# Saves our full report.
text_file = open("results.txt", "w")


text_file.write((classification_report(test_Y, predicted_classes, target_names=target_names)))

text_file.close()

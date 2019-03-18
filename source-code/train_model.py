# load modules

from keras import utils

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint

# settings

loss = 'categorical_crossentropy'

optimizer = 'adam'

validation_size = 5000

batch_size = 64

epochs = 10

# load Fashion MNIST training data set

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# normalize data to 0-1 range

x_train = x_train.astype('float32') / 255

# further break training data set into train and validation data sets

(x_train, x_validation) = x_train[validation_size:], x_train[:validation_size] 

(y_train, y_validation) = y_train[validation_size:], y_train[:validation_size]

# reshape input data from (width, height) to (width, height, 1)

width, height = x_train.shape[1], x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], width, height, 1)

x_validation = x_validation.reshape(x_validation.shape[0], width, height, 1)

# one-hot encode the labels

output_units = len(set(y_train));

y_train = utils.to_categorical(y_train, output_units)

y_validation = utils.to_categorical(y_validation, output_units)

# print training, validation data set shapes

print("x-train shape: ", x_train.shape)

print("y-train shape: ", y_train.shape)

print("x-validation shape: ", x_validation.shape)

print("y-validation shape: ", y_validation.shape)

# create the model

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(width, height, 1))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_units, activation='softmax'))

# take a look at the model summary

model.summary()

# compile the model

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# create model checkpoint

checkpointer = ModelCheckpoint(filepath='model-weights.hdf5', verbose=1, save_best_only=True)

# train the model

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_validation, y_validation), callbacks=[checkpointer])
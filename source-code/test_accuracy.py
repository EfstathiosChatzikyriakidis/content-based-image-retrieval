# load modules

from keras import utils

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# settings

loss = 'categorical_crossentropy'

optimizer = 'adam'

# load Fashion MNIST testing data set

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# normalize data to 0-1 range

x_test = x_test.astype('float32') / 255

# reshape input data from (width, height) to (width, height, 1)

width, height = x_test.shape[1], x_test.shape[2]

x_test = x_test.reshape(x_test.shape[0], width, height, 1)

# one-hot encode the labels

output_units = len(set(y_test));

y_test = utils.to_categorical(y_test, output_units)

# print testing data set shapes

print("x-test shape: ", x_test.shape)

print("y-test shape: ", y_test.shape)

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

# load the model

model.load_weights('model-weights.hdf5')

# evaluate the model on testing data set

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score[0])

print('Test accuracy: ', score[1])
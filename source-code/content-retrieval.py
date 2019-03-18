# load modules

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from keras import Model

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from sklearn.neighbors import NearestNeighbors

from sklearn.utils import shuffle

# settings

loss = 'categorical_crossentropy'

optimizer = 'adam'

n_queries = 10

layer = 'dense_one'

metric = 'cosine'

figure_size = (18, 7)

n_neighbors = 30

# define the class labels

class_labels = [ "T-shirt/top", # index 0
                 "Trouser",     # index 1
                 "Pullover",    # index 2 
                 "Dress",       # index 3 
                 "Coat",        # index 4
                 "Sandal",      # index 5
                 "Shirt",       # index 6 
                 "Sneaker",     # index 7 
                 "Bag",         # index 8 
                 "Ankle boot" ] # index 9

# load Fashion MNIST training, testing data set

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# randomize testing data and get a sample of it

x_test, y_test = shuffle(x_test, y_test, n_samples=n_queries)

# normalize data to 0-1 range

pixel_maximum_value = 255

x_train = x_train.astype('float32') / pixel_maximum_value

x_test = x_test.astype('float32') / pixel_maximum_value

# reshape input data from (width, height) to (width, height, 1)

width, height = x_train.shape[1], x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], width, height, 1)

x_test = x_test.reshape(x_test.shape[0], width, height, 1)

# calculate number of classes

n_classes = len(class_labels);

# create the model

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(width, height, 1))) 
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten(name="flatten_one"))
model.add(Dense(256, activation='relu', name="dense_one"))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

# compile the model

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# load the model

model.load_weights('model-weights.hdf5')

# use some search queries and get results

representation_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)

features_train = representation_model.predict(x_train)

features_test = representation_model.predict(x_test)

nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

nn.fit(features_train)

indexes = nn.kneighbors(X=features_test, return_distance=False)

x_train = x_train.reshape(x_train.shape[0], width, height)

x_test = x_test.reshape(x_test.shape[0], width, height)

for i, image in enumerate(x_test):
    plt.figure(figsize=(1.5, 1.5))

    plt.imshow(image, interpolation="bilinear", cmap=cm.gray)

    plt.title(class_labels[y_test[i]])

    plt.axis('off')

    plt.figure(figsize=figure_size)

    for j in range(n_neighbors):
        plt.subplot(n_neighbors / 10 + 1, 10, j + 1)

        plt.imshow(x_train[indexes[i][j]], interpolation="bilinear", cmap=cm.gray)

        plt.axis('off')
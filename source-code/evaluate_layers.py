# load modules

import csv

from collections import Counter

from keras import Model

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from sklearn.neighbors import NearestNeighbors

from sklearn.utils import shuffle

# settings

loss = 'categorical_crossentropy'

optimizer = 'adam'

n_queries = 100

layers = [ 'flatten_one', 'dense_one' ]

metrics = [ 'cosine' ]

n_neighbors_range_parameters = [ 1, 501, 1 ]

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

# calculate classes size

classes_size = Counter(y_train)

# calculate number of classes

n_classes = len(classes_size);

# print training, test data set shapes

print("x-train shape: ", x_train.shape)

print("y-train shape: ", y_train.shape)

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

model.add(Flatten(name="flatten_one"))
model.add(Dense(256, activation='relu', name="dense_one"))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

# take a look at the model summary

model.summary()

# compile the model

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# load the model

model.load_weights('model-weights.hdf5')

# functions

def evaluate_layer (nn, x_test, y_test, y_train, classes_size):
    elements_size = len(x_test)

    model_params = nn.get_params()

    results = nn.kneighbors(X=x_test, return_distance=False)

    sum_recall = 0

    sum_precision = 0

    for i, result in enumerate(results):
        element_label = y_test[i]

        labels = [y_train[index] for index in result]

        true_positives = labels.count(element_label);

        sum_precision += (true_positives / model_params['n_neighbors'])

        sum_recall += (true_positives / classes_size[element_label])

    avg_precision = sum_precision / elements_size
    
    avg_recall = sum_recall / elements_size

    return (avg_precision, avg_recall)

# evaluate layers and get results

results = []

for layer in layers:
    representation_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)

    features_train = representation_model.predict(x_train)

    features_test = representation_model.predict(x_test)

    for metric in metrics:
        for k in range(*n_neighbors_range_parameters):
            nn = NearestNeighbors(n_neighbors=k, metric=metric)

            nn.fit(features_train)

            avg_precision, avg_recall = evaluate_layer (nn, features_test, y_test, y_train, classes_size)

            results.append([layer, metric, k, "%.3f" % (100 * avg_precision), "%.3f" % (100 * avg_recall)])

# store the results

with open('precision-recall-results.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerow(['layer', 'metric', 'k', 'avg precision (%)', 'avg recall (%)'])
    writer.writerows(results)
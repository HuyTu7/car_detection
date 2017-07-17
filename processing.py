from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import metrics
import logging
import json
import csv
import numpy as np
import pandas as pd
import time
np.random.seed(100)

import utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
import threading
import Queue
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)

class Dataset:
    def __init__(self, dump_path, csv_path, is_dumped=True):
        self.raw_data = raw_data = pd.read_csv(csv_path)
        raw_data.info()
        if not is_dumped:
            queue = Queue.Queue()
            for index, row in raw_data.iterrows():
                t = threading.Thread(target=utils.load_image,
                                     args=(row['labels'], row['files'], queue,))
                t.daemon = True
                t.start()
            queue.join()
            item = queue.get()
            self.X = item['data']
            self.Y = item['y']
            while not queue.empty():
                item = queue.get()
                self.X = np.vstack((self.X, item['data']))
                self.Y = np.vstack((self.Y, item['y']))
            self.X.dump(dump_path + '/data_x.numpy')
            self.Y.dump(dump_path + '/data_y.numpy')
        else:
            self.X = np.load(dump_path + '/data_x.numpy')
            self.Y = np.load(dump_path + '/data_y.numpy')
            self.Y = np_utils.to_categorical(self.Y)
        logging.info("shape of train data: %s" % str(self.X.shape))
        logging.info("Load data Done!")

    def getTrainTest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y,
                                                            test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test


class CarModel:
    def __init__(self, load=None):
        if not load:
            self.model = model = Sequential()
            print "hi"
            model.add(Conv2D(32, (3, 3), input_shape=(40, 100, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
            model.add(Dropout(0.4))
            model.add(Dense(2, activation='softmax'))
            '''
            model.add(Conv2D(48, (3, 3), padding='same', activation='relu', input_shape=(3, 40, 40)))
            model.add(Dropout(0.2))
            model.add(Conv2D(48, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(96, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(192, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(512), activation='relu')
            model.add(Dropout(0.5))
            model.add(Dense(256), activation='relu')
            model.add(Dropout(0.5))
            model.add(Dense(2, activation='softmax'))
            '''
        else:
            from keras.models import model_from_json
            json_file = open(load.rsplit('.', 1)[0] + '.json')
            self.model = model_from_json(json_file.read())
            json_file.close()
            # load weight
            self.model.load_weights(load)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print model.summary()

    def train(self, train_X, train_Y, test_X, test_Y):
        # epochs, batch_size
        train_datagen = ImageDataGenerator(rescale= 1./255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_datagen.fit(train_X)
        test_datagen.fit(test_X)
        steps_train = 2000 // 32
        steps_test = len(test_X) / 32
        # Train the model
        logging.info("Train data: ")
        logging.info("  train shape:  " + str(train_X.shape))
        logging.info("  test shape:  " + str(test_X.shape))
        logging.info("Model is training....")
        start = time.time()
        self.model.fit_generator(
            train_datagen.flow(train_X, train_Y, shuffle=True),
            steps_per_epoch=steps_train,
            epochs=50,
            validation_data=test_datagen.flow(test_X, test_Y, shuffle=True),
            validation_steps=steps_test
        )
        end = time.time()
        # save the trained model
        # serialize model to JSON
        model_json = self.model.to_json()

        with open("./models/car.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("./models/car.h5")

        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps=steps_test)
        print "Model took %0.2f seconds to train" % (end - start)
        print scores
        print "%s: %.3f%%" % (self.model.metrics_names[1], scores[1])

    def getEvaluate(self, test_X, test_Y):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen.fit(test_X)
        steps = len(test_X) / 32
        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps=steps)
        print "%s: %.3f%%" % (self.model.metrics_names[1], scores[1] )

    def predict(self, image, preprocessed=True):
        """
        image is a 3D array, if image is preprocessed, scale and resize it
        if img isnt preprocessed, image is a pathlink.
        """
        if not preprocessed:
            image = utils.LoadandPreprocessImg(image)
        else:
            image /= 255.
            image = np.resize(image, (1, 40, 40, 3))
        score = self.model.predict(image)
        return score

    def batchPredict(self, batch_image, preprocessed=True):
        pass


def get_labels():
    labels = dict()
    for i in range(550):
        if i < 500:
            labels["./CarDataset/TrainImages/neg-%s.pgm" % i] = 0
        labels["./CarDataset/TrainImages/pos-%s.pgm" % i] = 1
    with open('car_labels.json', 'w') as fp:
        json.dump(labels, fp)
    with open('car_labels.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in labels.items():
            writer.writerow([key, value])
    return labels


if __name__ == '__main__':
    print "Load models"
    dataset = Dataset('./CarDataset', './CarDataset/car_labels.csv')
    model = CarModel()
    #model.model.load_weights('./models/50beauty.h5')
    train_x, train_y, test_x, test_y = dataset.getTrainTest()
    #print train_y      
    model.train(train_x, train_y, test_x, test_y)
    model.getEvaluate(test_x, test_y)
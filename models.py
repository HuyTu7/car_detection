from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import metrics
import logging

import numpy as np
import pandas as pd
np.random.seed(100)

import utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
import threading
import Queue
import math

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                        level=logging.INFO)
class Dataset:
    def __init__(self, dump_path, csv_path, is_dumped=True):
        self.raw_data = raw_data = pd.read_csv(csv_path) 
        raw_data.info()
        if not is_dumped:
            queue = Queue.Queue()
            for index, row in raw_data.iterrows():
                t = threading.Thread(target=utils.load_image,\
                args=(row['Attractiveness label'],row['Files'], queue,))
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
           
            self.Y /= 5.0
            self.X.dump(dump_path)
            self.Y.dump(dump_path)
        else:
            self.X = np.load(dump_path+'/data_x.numpy')
            self.Y = np.load(dump_path+'/data_y.numpy')
        logging.info("shape of train data: %s"%str(self.X.shape)) 
        logging.info("Load data Done!")
    def getTrainTest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, \
                test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test
class BeautyModel:
    def __init__(self, load=None):
        if not load:
            self.model = model = Sequential()

            # model.add(Conv2D(50, (5, 5), input_shape=(227, 227, 3), activation='relu',\
            # kernel_constraint=maxnorm, padding='same'))
            model.add(Conv2D(50, (5, 5), input_shape=(227, 227, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(100, (5, 5), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(150, (4, 4), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(200, (4, 4), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))

            model.add(Conv2D(250, (4, 4), activation='relu', padding='same'))        
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(300, (2, 2), activation='relu', padding='same'))        
            model.add(AveragePooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(500, activation='relu', kernel_constraint=maxnorm(3)))
            
            model.add(Dense(1, activation='sigmoid'))
        else:
            from keras.models import model_from_json
            json_file = open(load.rsplit('.', 1)[0] + '.json')
            self.model = model_from_json(json_file.read())
            json_file.close()
            #load weight
            self.model.load_weights(load)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        
        #model.summary()
    def train(self, train_X, train_Y, test_X, test_Y):
        #epochs, batch_size
        train_datagen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_datagen.fit(train_X)
        test_datagen.fit(test_X)
        steps_train = 2000 // 32
        steps_test = len(test_X)/32

        logging.info("Train data: ")
        logging.info("  train shape:  " + str(train_X.shape))
        logging.info("  test shape:  " + str(test_X.shape))
        logging.info("Model is training....")
        self.model.fit_generator(
            train_datagen.flow(train_X, train_Y, shuffle=True),
            steps_per_epoch=steps_train,
            epochs=100,
            validation_data=test_datagen.flow(test_X, test_Y, shuffle=True),
            validation_steps=steps_test
        )
        # save the trained model
        # serialize model to JSON
        model_json = self.model.to_json()

        with open("./models/beauty.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("./models/beauty.h5")

        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps = steps_test)
        print scores
        print "%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100)
    def getEvaluate(self, test_X, test_Y):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen.fit(test_X)
        steps = len(test_X)/32
        scores = self.model.evaluate_generator(test_datagen.flow(test_X, test_Y), steps=steps)
        print "%s: %.2f%%" % (self.model.metrics_names[1], (1-math.sqrt(scores[1]))*100)

    def predict(self, image, preprocessed=True):
        """
        image is a 3D array, if image is preprocessed, scale and resize it
        if img isnt preprocessed, image is a pathlink.
        """
        if not preprocessed:
            image = utils.LoadandPreprocessImg(image)
        else:
            image /= 255.
            image = np.resize(image, (1, 227, 227 ,3))
        score = self.model.predict(image)
        
        return score
    def batchPredict(self, batch_image, preprocessed=True):
        pass
if __name__ == '__main__':
    print "load models"
    dataset = Dataset('./dataset', './dataset/SCUT_FBP.csv')
    model = BeautyModel()
    model.model.load_weights('./models/50beauty.h5')

    train_x, train_y, test_x, test_y = dataset.getTrainTest()
    model.train(train_x, train_y, test_x, test_y)
    model.getEvaluate(test_x, test_y)
    """example data to test the model is working"""

import os
import cv2
import yaml
import time
import pickle
import logging
import warnings
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger()
warnings.filterwarnings('ignore')


class Preprocess:
    """Preprocess: Class for processing the dataset into features and labels"""

    def __init__(self):
        """initialises the config variables and load datasets"""

        with open('./_config.yml', 'r') as outfile:
            try:
                self.yml = yaml.load(outfile)
            
            except yaml.YAMLError as error:
                logger.error('[/!/] _config.yml file not found:', error)
        
        try:
            self.Xdata = np.load('./'+ self.yml['_dataset'] +'/X.npy')
            self.Ydata = np.load('./'+ self.yml['_dataset'] +'/Y.npy')
        
        except FileNotFoundError:
            logger.error('[/!/] Dataset files not found in folder!')

    def processing(self):
        """process the numpy matrix into features and labels; Into training and testing sets"""

        Y = np.array([])
        logging.debug('[.] Converting label matrix into one-digit form ... e.g. [0. 0. 1. 0. 0. 0. 0. 0. 0.] -> 2.')
        
        for series in self.Ydata:
            for i in range(0, len(series)):
                
                if series[i] == 1.0:
                    Y = np.append(Y, i)

        Y = Y.reshape(self.Xdata.shape[0], 1)
        X_train, X_test, Y_train, Y_test = train_test_split(self.Xdata, Y, test_size=0.15, random_state=42)

        no_train = X_train.shape[0]
        no_test = X_test.shape[0]

        logging.debug('[#] Data matrix shapes changed')

        X_train_flatten = X_train.reshape(no_train, X_train.shape[1]*X_train.shape[2])
        X_test_flatten = X_test.reshape(no_test, X_test.shape[1]*X_test.shape[2])

        logging.info('[.] Dataset processed')
        logging.debug('[#] Dictionary returned with training and testing data')

        return {"X_train": X_train_flatten.T,
                "Y_train": Y_train.T,
                "X_test": X_test_flatten.T,
                "Y_test": Y_test.T}


class Model:
    """Model: Class for training, testing, and prediction of input by deep learning model"""

    def __init__(self):
        """initialising -> null variables"""

        with open('./_config.yml', 'r') as outfile:
            try:
                self.yml = yaml.load(outfile)

            except yaml.YAMLError as error:
                logger.error('[/!/] _config.yml file not found:', error)

        self.kwargs = self.yml['_modeldict'] 


    def model(self):
        """dataset called and trained the model with deep learning algorithms"""

        self.pre = Preprocess()
        self.datadict = self.pre.processing()

        logging.info('[.] Forming deep learning model ...')
        logging.debug('[#] 150 iterations choosed! ')

        if self.kwargs['forest']:
            model = RandomForestClassifier(n_estimators=15, random_state=42)
        elif self.kwargs['logistic']:
            model = LogisticRegression(random_state=42, max_iter=150)


        model = model.fit(self.datadict["X_train"].T, self.datadict["Y_train"].T)
        
        accuracy = model.score(self.datadict["X_test"].T, self.datadict["Y_test"].T)
        
        print('-------------Model-------------')
        print("Accuracy on training set: {:.4f} %".format(model.score(self.datadict["X_train"].T, self.datadict["Y_train"].T)*100))
        print("Accuracy on testing set: {:.4f} %".format(accuracy*100))
       
        logging.info('[.] Saving deep learning model ...')
        
        if self.kwargs['logistic']:
            pickle.dump(model, open('./_models/_modellogistic', 'wb'))
        else:
            pickle.dump(model, open('./_models/_modelforest', 'wb'))

        return True

    def prediction(self, filepath):
        """predicting the input image with already trained model in `_model`"""

        logging.info('[.] Image processing with PIL ...')
        logging.debug('[#] converting in mode \'1\' and shape 4096*1 ')

        img = Image.open(filepath).convert('L').resize((64,64), Image.ANTIALIAS)
        data = np.asarray(img.getdata()).reshape(1, -1)

        try:
            if self.kwargs['logistic']:
                loaded_model = pickle.load(open('./_models/_modellogistic', 'rb'))
            else:
                loaded_model = pickle.load(open('./_models/_modelforest', 'rb'))
        except Exception as e:
            logging.error('[/!/] Seems like model isn\'t rained, please train the model first!')

        predict = loaded_model.predict(data)

        print(predict)
        return True


    def predictlive(self):
        """Getting the image from webcam and then predicting it based on already trained model"""
        
        camera = cv2.VideoCapture(0)

        logger.warning('[!] Webcam started ...')
        r, img = camera.read()
        time.sleep(1)
        r, img1 = camera.read()

        logging.info('[.] Photo clicked and saved.')
        cv2.imwrite('./signimage.png', img1)
        self.prediction('./signimage.png')

        os.remove('./signimage.png')
        logging.info('[.] Deleted the saved photo')

        logging.warning('[!] Webcam dead. ')
        del camera


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Sign Langauge Digit Classification')

    parser.add_argument('-t', '--train', help='Train the model', action="store_true", default=False)
    parser.add_argument('-p', '--predict', help='Path to the image to be predicted', default='')    
    parser.add_argument('-cp', '--clickP', help='Click picture through webcam and predict', action="store_true", default=False)

    parser.add_argument('-L', '--log', help='Set the logging level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
        
    args = parser.parse_args()
    logging.basicConfig(level=args.log)

    mod = Model()

    if args.train:
        mod.model()

    if args.predict is not '':
        mod.prediction(args.predict)

    if args.clickP:
        print('[*] Please form a sign-language digit in front of the webcam (at a proper bright place).')
        mod.predictlive()

    print('[*] Created by {}.'.format(mod.pre.yml['_creator']))
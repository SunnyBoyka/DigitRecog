import yaml
import pickle
import logging
import warnings
import argparse

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from PIL import Image
import matplotlib.pyplot as plt


logger = logging.getLogger()
warnings.filterwarnings('ignore')


class Preprocess:
    def __init__(self):

        with open('./_config.yml', 'r') as outfile:
            try:
                self.yml = yaml.load(outfile)
            
            except yaml.YAMLError as error:
                logger.error('[/!/] _config.yml file not found:', error)
        
        try:
            self.Xdata = np.load('./dataset/X.npy')
            self.Ydata = np.load('./dataset/Y.npy')
        
        except FileNotFoundError:
            logger.error('[/!/] Dataset files not found in folder!')


    def flatten(self):

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
    def __init__(self):
        pass

    def model(self):
        datadict = Preprocess().flatten()

        logging.info('[.] Forming deep learning model ...')
        logging.debug('[#] 150 iterations choosed! ')

        logreg = linear_model.LogisticRegression(random_state=42, max_iter=150)
        logreg = logreg.fit(datadict["X_train"].T, datadict["Y_train"].T)
        
        accuracy = logreg.score(datadict["X_test"].T, datadict["Y_test"].T)
        
        print("Accuracy on training set: {:.4f} %".format(logreg.score(datadict["X_train"].T, datadict["Y_train"].T)*100))
        print("Accuracy on testing set: {:.4f} %".format(accuracy*100))

        logging.info('[.] Saving deep learning model ...')
        pickle.dump(logreg, open('_model', 'wb'))

        return True


    def prediction(self, filepath):

        logging.info('[.] Image processing with PIL ...')
        logging.debug('[#] converting in mode \'1\' and shape 4096*1 ')

        img = Image.open(filepath).convert('L').resize((64,64), Image.ANTIALIAS)
        data = np.asarray(img.getdata()).reshape(1, -1)

        loaded_model = pickle.load(open('_model', 'rb'))
        predict = loaded_model.predict(data)

        print(predict)
        return True


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Sign Langauge Digit Classification')

    parser.add_argument('-path', help='Path to the image to be predicted', default='')    
    parser.add_argument('-L', '--log', help='Set the logging level', type=str, choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'])
        
    args = parser.parse_args()
    logging.basicConfig(level=args.log)

    mod = Model()
    if args.path is not '':
        mod.prediction(args.path)
    else:
        mod.model()

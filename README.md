# Sign-Language-Digits
Using deep learning to classify sign language digits.

## Sign Langauge
[Sign languages](https://en.wikipedia.org/wiki/Sign_language) are languages that use manual communication to convey meaning. This can include simultaneously employing hand gestures, movement, orientation of the fingers, arms or body, and facial expressions to convey a speaker's ideas. Similarly, we can employ hand gestures to convey the meaning of digits. We will use machine learning to train the models with processed datasets, and then predict for new images. 

### Dataset Preview:
|<img src="images/example_0.JPG">|<img src="images/example_1.JPG">|<img src="images/example_2.JPG">|<img src="images/example_3.JPG">|<img src="images/example_4.JPG">|
|:-:|:-:|:-:|:-:|:-:|
|0|1|2|3|4|
|<img src="images/example_5.JPG">|<img src="images/example_6.JPG">|<img src="images/example_7.JPG">|<img src="images/example_8.JPG">|<img src="images/example_9.JPG">|
|5|6|7|8|9|

## Performance
#### Logistic Regression
```console
gavy42@jarvis:~/Sign-Language-Digits$ python3 model.py -t
-------------Model-------------
Accuracy on training set: 99.5434 %
Accuracy on testing set: 73.8710 %
```

#### Random Forest
```console
gavy42@jarvis:~/Sign-Language-Digits$ python3 model.py -t
-------------Model-------------
Accuracy on training set: 100.0000 %
Accuracy on testing set: 73.8710 %

```

## Usage
1. Clone the repository as `git clone https://github.com/techcentaur/Sign-Language-Digits.git`.
2. Run `pip3 install -r requirements.txt` to install the dependencies.
3. Run `python3 model.py -t` to train the model
> Run `python3 model.py -cp` to click the picture of sign digit from webcam, and then do the prediction.


### Argparse Usage
```console
gavy42@jarvis:~/Sign-Language-Digits$ python3 model.py -h
usage: model.py [-h] [-t] [-p PREDICT] [-cp]
                [-L {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Sign Langauge Digit Classification

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           Train the model
  -p PREDICT, --predict PREDICT
                        Path to the image to be predicted
  -cp, --clickP         Click picture through webcam and predict
  -L {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level
```

### Functional Usage
The main and only python file is `model.py`. It has two classes.
- `Preprocess`: Class for processing the dataset into features and labels
	- `processing()`: Returns dict as ```{"X_train": <np.array>, "Y_train": <np.array>, "X_test": <np.array>, "Y_test": <np.array>}```; This function process the numpy matrix into features and labels; Into training and testing sets.
- `Model`: Class for training, testing, and prediction of input by deep learning model
	- `model()`: Training and testing of machine leaning model.
	- `prediction(<filepath>)`: Predicting the input image with already trained model saved in `_model`; Prints the predicted digit.
	- ```predictlive(self)```: Getting the image from webcam and then predicting it based on already trained model, i.e., calling `prediction()` function.


## Configuration
Edit `_config.yml` here for changing learning model or dataset folder name.

```yml
# Dataset foldername
_dataset: dataset
# Model for learning
_modeldict: {"logistic": False, "neuralnet": False, "forest": True}
```

## About Dataset
I got the dataset from [this](https://github.com/ardamavi/Sign-Language-Digits-Dataset) GitHub repository. Cheers to that.

## Motivation
I was looking forward to use deep learning on manipulative datasets and found this dataset. It was interesting and useful.

## Support
- If you have any problem regarding the code and need help, feel free to raise an issue or drop a mail at `ankit03june@gmail.com`.
- Pull requests are more than welcome.

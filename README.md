# Sign-Language-Digits
Using deep learning to classify sign language digits.

## Sign Langauge
[Sign languages](https://en.wikipedia.org/wiki/Sign_language) are languages that use manual communication to convey meaning. This can include simultaneously employing hand gestures, movement, orientation of the fingers, arms or body, and facial expressions to convey a speaker's ideas.

## Performance

#### Logistic Regression
```console
gavy42@jarvis:~/Sign-Language-Digits$ python3 model.py -t
-------------Model-------------
Accuracy on training set: 99.5434 %
Accuracy on testing set: 73.8710 %

```

## Usage
1. Clone the repository as `git clone `.
2. Run `pip3 install -r requirements.txt` to install the dependencies.
3. Run `python3 model.py` to run the model

### Argparse Usage
```console
gavy42@jarvis:~/Sign-Language-Digits$ python3 model.py -h
usage: model.py [-h] [-t] [-p PREDICT]
                [-L {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Sign Langauge Digit Classification

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           Train the model
  -p PREDICT, --predict PREDICT
                        Path to the image to be predicted
  -L {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level

```

### Functional Usage


## About Dataset
I got the dataset from [this](https://github.com/ardamavi/Sign-Language-Digits-Dataset) GitHub repository. Thanks to that.

## Motivation
I was looking forward to use deep learning on manipulative datasets and found this dataset. It was interesting, and useful

## Support
If you have any problem regarding the code and need help, feel free to raise an issue or drop a mail [here](mailto: ankit03june@gmail.com).

Pull requests are welcome.

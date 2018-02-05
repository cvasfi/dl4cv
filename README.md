# Facial Emotion Recognition using CNN

The aim is to build and test CNNs to tackle the FER (Facial Emotion Recognition) problem. 

## Dataset
The project expects the fer2013 dataset to be present as a csv file named `fer2013.csv` under `./data/` directory.

1. Download the dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. Untar the file and place the fer2013.csv fil under the ./data/ directory

## Dependencies
The project makes use of Python 2.7 and PyTorch. 

PyTorch can installed from [here](http://pytorch.org/)

Other dependencies can be found in the `requirements.txt` file

Run the following command in the project home to install the python libraries.
```
pip install -r ./requirements.txt
```
## Running the code

The network can be trained by running the `train.py` in the home directory of the project.

The architecture of the network can be specified through the --model parameter. An example is shown below.

```
python train.py --model resnet20
```
Valid models include `bkvgg12`, `resnet20` and `cnn_sift`

The list of all arguments that can be provided to `train.py`can be see by running the following command

```
python train.py --help
```
There are arguments that can be used to specify the epcohs, batch_size, optmizer etc.

## Optional Dependencies
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) (option)
* [TensorFlow](https://www.tensorflow.org/) (option)



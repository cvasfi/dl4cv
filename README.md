# dl4cv project

The aim is to build and test networks to improve the accuracy on kaggle emotion dataset

## DataSet
The project expects the fer2013 dataset to be present as a csv file named `fer2013.csv` under `./data/` directory.

* Download the dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
* Untar the file and place the fer2013.csv fil under the ./data/ directory

## Dependencies
The project makes use of Python 2.7 and PyTorch. PyTorch can installed from [here](http://pytorch.org/)

Other dependencies can be found in the `requirments.txt` file

Run the following command in the project home to install the python libraries.
```
pip install -r ./requirements.txt
```
## Optional Dependencies
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) (option)
* [TensorFlow](https://www.tensorflow.org/) (option)


## Demo
```
python train.py --model resnet20
```


# dl4cv project

The aim is to build and test networks to improve the accuracy on kaggle emotion dataset

## Dependencies
* Python 2.7
* [PyTorch](http://pytorch.org/)
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) (option)
* [TensorFlow](https://www.tensorflow.org/) (option)

## Demo
```
python train.py --model resnet20
```

## DataSet
The project expects the fer2013 dataset to be present as a csv file named `fer2013.csv` under `./data/` directory.
* Download the dataset from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
* Untar the file and place the fer2013.csv fil under the ./data/ directory


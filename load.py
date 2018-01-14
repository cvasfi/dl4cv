import os
import numpy as np
import pandas as pd
import cv2 as cv

facial_expressions = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")

def load_data_from_csv(path="data/fer2013.csv", filter_dataset=False):
    """
    Load data from kaggle csv file and parse to numpy arrays ready for training
    :param path: a relative path to the csv file from the project directory
    :param filter_dataset: a flag whether to filter the dataset or not
    :return: train_x: numpy array
    :return: train_y: numpy array
    :return: test_x: numpy array
    :return: test_y: numpy array
    """
    path = os.path.join(os.path.dirname(__file__), path)
    assert os.path.isfile(path)
    df = pd.read_csv(path)

    # filter the data
    if filter_dataset:
        bad_data_indices = get_bad_samples()
        if bad_data_indices is not None:
            df = df.drop(df.index[bad_data_indices.tolist()])

    # Data Transformation
    train_x = df.loc[df['Usage'].isin(['Training', 'PublicTest']), ['pixels']]
    train_y = df.loc[df['Usage'].isin(['Training', 'PublicTest']), ['emotion']]
    train_x = transform_x(train_x)
    train_y = transform_y(train_y)

    test_x = df.loc[df['Usage'] == 'PrivateTest', ['pixels']]
    test_y = df.loc[df['Usage'] == 'PrivateTest', ['emotion']]
    test_x = transform_x(test_x)
    test_y = transform_y(test_y)
    return train_x, train_y, test_x, test_y


def save_data_to_npy(train_x, train_y, test_x, test_y, path="data"):
    """
    Write train_x, train_y, test_x, test_y to disk in npy format for faster loading and no parsing overhead.
    :param train_x: numpy array
    :param train_y: numpy array
    :param test_x: numpy array
    :param test_y: numpy array
    :param path: relative path to the project directory
    """
    path = os.path.join(os.path.dirname(__file__), path)
    np.save(os.path.join(path, 'train_x'), train_x)
    np.save(os.path.join(path, 'train_y'), train_y)
    np.save(os.path.join(path, 'test_x'), test_x)
    np.save(os.path.join(path, 'test_y'), test_y)


def load_data_from_npy(path="data"):
    """
    load the fer dataset from .npy files
    :param path: relative path to the project directory where the npy files are
    :return: train_x: numpy array
    :return: train_y: numpy array
    :return: test_x: numpy array
    :return: test_y: numpy array
    """
    path = os.path.join(os.path.dirname(__file__), path)
    train_x = np.load(os.path.join(path, 'train_x.npy'))
    train_y = np.load(os.path.join(path, 'train_y.npy'))
    test_x = np.load(os.path.join(path, 'test_x.npy'))
    test_y = np.load(os.path.join(path, 'test_y.npy'))
    return train_x, train_y, test_x, test_y


def remove_npy_files(path="data"):
    """
    delete the npy files in the path provided
    :param path: relative path to the project directory where the npy files are
    """
    path = os.path.join(os.path.dirname(__file__), path)
    try:
        os.remove(os.path.join(path, 'train_x.npy'))
        os.remove(os.path.join(path, 'train_y.npy'))
        os.remove(os.path.join(path, 'test_x.npy'))
        os.remove(os.path.join(path, 'test_y.npy'))
    except OSError:
        pass


def get_bad_samples(path="data/badtrainingdata.txt"):
    """
    Read the bad training data file to filter the dataset.
    This function reads a file that is located by default at data dir.
    Every line in the file has an index of a bad sample in the csv file.
    :param path:
    :return:
    """
    file_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.isfile(file_path):
        with open(file_path) as file:
            return np.array([int(line.strip()) for line in file.readlines()
                             if not line.isspace() and line.strip().isnumeric()])
    else:
        return None


def check_if_data_available_in_npy(path="data"):
    """
    check if the data is present in numpy .npy format
    :param path: relative path to the current working directory
    :return: True if is all data files are available, False otherwise
    """
    path = os.path.join(os.path.dirname(__file__), path)
    return os.path.isfile(os.path.join(path, 'train_x.npy')) and os.path.isfile(os.path.join(path, 'train_y.npy')) and \
           os.path.isfile(os.path.join(path, 'test_x.npy')) and os.path.isfile(os.path.join(path, 'test_y.npy'))


def transform_x(data_frame):
    """
    transform feature data to compatible shape with the keras model
    :param data_frame: panda data frame
    :return: data_frame: numpy array
    """
    data_frame = data_frame['pixels']  # Selecting Pixels Only
    data_frame = data_frame.values  # Converting from Panda Series to Numpy Ndarray
    data_frame = data_frame.reshape((data_frame.shape[0], 1))  # Reshape for the subsequent operation
    # convert pixels from string to ndarray
    data_frame = np.apply_along_axis(lambda x: np.array(x[0].split()).astype(dtype=float), 1, data_frame)
    data_frame = data_frame.reshape((data_frame.shape[0], 48, 48, 1))  # reshape to NxHxWxC
    return data_frame


def transform_y(data_frame):
    """
    transform target data to compatible shape with keras model
    :param data_frame: panda data frame with target columns
    :return: data_frame: Numpy array of shape (N * number of classes)
    """
    data_frame = data_frame['emotion']  # Selecting Emotion Only
    data_frame = data_frame.astype('category', categories=list(range(7)))
    data_frame = pd.get_dummies(data_frame)
    data_frame = data_frame.values
    return data_frame


def visualize_data(path="data/fer2013.csv"):
    """
    Display images from the dataset one by one.
    :param path: a relative path to the csv file from the project directory
    """
    path = os.path.join(os.path.dirname(__file__), path)
    assert os.path.isfile(path)
    df = pd.read_csv(path)
    for pixels in df['pixels']:
        cv.namedWindow("FER images", cv.WINDOW_NORMAL)
        cv.imshow("FER images", cv.resize(np.array(pixels.split(), dtype=np.uint8).reshape(48, 48), (768, 768)))
        if cv.waitKey(0) & 0xFF == ord('q'):
            break


def visualize_bad_data(path="data/fer2013.csv"):
    """
    Display bad images in the dataset.
    :param path: a relative path to the csv file from the project directory
    """
    path = os.path.join(os.path.dirname(__file__), path)
    assert os.path.isfile(path)
    df = pd.read_csv(path)
    bad_data_indices = get_bad_samples()
    bad_data_indices = [] if bad_data_indices is None else bad_data_indices.tolist()
    for index in bad_data_indices:
        cv.namedWindow("FER bad images", cv.WINDOW_NORMAL)
        cv.imshow("FER bad images", cv.resize(np.array(df.iloc[index]['pixels'].split(), dtype=np.uint8).
                                              reshape(48, 48), (768, 768)))
        if cv.waitKey(0) & 0xFF == ord('q'):
            break


def print_summary(train_y, test_y):
    """
    print summary about the data
    :param train_y: numpy array
    :param test_y: numpy array
    """
    print("---------------------------------Summary----------------------------------")
    print("Number of training samples: ", train_y.shape[0])
    print("Number of test samples: ", test_y.shape[0])
    print("Number of Facial Expressions: ", len(facial_expressions))
    print("Facial expressions: ", facial_expressions)
    print("Facial expressions counts: ")
    for index, count in enumerate(np.add(np.sum(train_y, axis=0), np.sum(test_y, axis=0)).tolist()):
        print(facial_expressions[index], ": ", count)
    print("--------------------------------------------------------------------------")


class DataLoader:
    @staticmethod
    def load_data(csv_path="data/fer2013.csv", filter_dataset=True):
        """
        load the data from npy files if available otherwise load from csv, filter the data and generate npy files.
        note that this method doesn't guarantee the data loaded will be updated using current filtering data.
        :param csv_path: a relative path to the csv file from the project directory
        :param filter_dataset: boolean flag whether to filter the data or not.
        :return: numpy arrays
        """
        if check_if_data_available_in_npy():
            train_x, train_y, test_x, test_y = load_data_from_npy()
        else:
            train_x, train_y, test_x, test_y = load_data_from_csv(csv_path, filter_dataset=filter_dataset)
            save_data_to_npy(train_x, train_y, test_x, test_y)
        return train_x, train_y, test_x, test_y

    @staticmethod
    def load_data_csv(csv_path="data/fer2013.csv", filter_dataset=True):
        """
        Load the data from csv and generate the npy files after filtering the data.
        note that this method guarantees the data will be updated using current filtering data.
        :param csv_path: a relative path to the csv file from the project directory
        :param filter_dataset: boolean flag whether to filter the data or not.
        :return: numpy arrays
        """
        remove_npy_files()
        train_x, train_y, test_x, test_y = load_data_from_csv(csv_path, filter_dataset=filter_dataset)
        save_data_to_npy(train_x, train_y, test_x, test_y)
        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # by default search for the data in npy format, if not, read the csv and save as npy for later runs
    if check_if_data_available_in_npy():
        x_train, y_train, x_test, y_test = load_data_from_npy()
    else:
        x_train, y_train, x_test, y_test = load_data_from_csv()
        save_data_to_npy(x_train, y_train, x_test, y_test)

    # print summary
    print_summary(y_train, y_test)

    # visualize the dataset
    # visualize_data()

    # visualize bad data in the dataset
visualize_bad_data()
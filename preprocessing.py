import time
import os
import pandas
import numpy as np
from collections import defaultdict
from scipy import sparse


class PreProcessing:

    # directory paths
    csv_path = 'csv/'
    dataset_path = 'dataset/'

    # filenames
    anime_filename = 'anime.csv'
    rating_filename = 'rating.csv'
    remapped_filename = 'remapped.csv'
    dataset_train_filename = 'data_train'
    dataset_test_filename = 'data_test'

    # filepaths
    movie_filepath = csv_path + anime_filename
    rating_filepath = csv_path + rating_filename
    remapped_filepath = csv_path + remapped_filename

    def __init__(self):
        # record start time
        time_start = time.time()
        # check whether the csv data exists
        if not (os.path.isfile(self.movie_filepath) and os.path.isfile(self.rating_filepath)):
            print('no csv data provided')
            exit(1)
        # ensure the dataset path exists
        if not os.path.isdir(self.dataset_path):
            os.mkdir(self.dataset_path)
        # load remapped data and generate remapped data if not exist
        if os.path.isfile(self.remapped_filepath):
            self.rating_data = self.__load_csv_ndarray__(self.remapped_filepath)
        else:
            print('no remapped csv found')
            self.rating_data = self.__remap_csv__(self.remapped_filepath)

        pass

    def __remap_csv__(self, target_filepath):
        print('remapping data', end='    ')
        time_start = time.time()
        # set up csv file headers
        movie_headers = ['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members']
        rating_headers = ['user_id', 'anime_id', 'rating']
        # read data using pandas
        movie_data = pandas.read_csv(self.movie_filepath, sep=',', names=movie_headers, header=None, skiprows=1)
        rating_data = pandas.read_csv(self.rating_filepath, sep=',', names=rating_headers, header=None, skiprows=1)
        # map movie id to row number in csv
        movie_mapping = defaultdict(lambda: np.NaN)
        for idx, row in movie_data.iterrows():
            movie_mapping[row[0]] = idx + 2
        # drop rows with NaN
        rating_data = rating_data.dropna()
        # drop rows with a rating of -1
        rating_data = rating_data[rating_data.rating != -1]
        # remapping
        rating_data['anime_id'] = rating_data['anime_id'].map(movie_mapping)
        # drop rows with movie_id NaN
        rating_data = rating_data.dropna(how='any')
        # set movie_id to integer type
        rating_data['anime_id'] = rating_data['anime_id'].astype(int)
        # export to csv
        rating_data.to_csv(target_filepath, columns=rating_headers, index=0)
        print('done in {} seconds'.format(time.time() - time_start))
        return rating_data.to_numpy()

    def split_data(self, random=True, percentage=0.8):
        print('splitting data {}'.format('randomly' if random else 'arbitrarily'), end='    ')
        time_start = time.time()
        # make a copy of rating data
        rating_data = self.rating_data.copy()
        # find the split index
        split_idx = int(percentage * len(rating_data[:, 0]))
        # if the method is random, then shuffle the data array first
        if random is True:
            np.random.shuffle(rating_data)
        # split data
        train_data, test_data = rating_data[:split_idx, :], rating_data[split_idx:, :]
        train_user, train_movie, train_rating = train_data.T
        test_user, test_movie, test_rating = test_data.T
        # find matrix dimensions
        user_size = int(max(rating_data[:, 0]) + 1)
        movie_size = int(max(rating_data[:, 1]) + 1)
        # build data matrices (in coo format)
        matrix_train = sparse.coo_matrix((train_rating, (train_movie, train_user)),
                                         shape=(movie_size, user_size), dtype=np.float64)
        matrix_test = sparse.coo_matrix((test_rating, (test_movie, test_user)),
                                        shape=(movie_size, user_size), dtype=np.float64)
        print('done in {} seconds'.format(time.time() - time_start))
        # save split data to npz file
        self.__save_data_matrix_npz__(matrix_train, matrix_test)

    def __load_csv_ndarray__(self, csv_path):
        print('loading remapped data', end='    ')
        time_start = time.time()
        data = pandas.read_csv(csv_path, sep=',', names=None, header=None, skiprows=1).to_numpy()
        print('done in {} seconds'.format(time.time() - time_start))
        return data

    def __save_data_matrix_npz__(self, matrix_train, matrix_test, suffix=''):
        train_file = self.dataset_path + self.dataset_train_filename + suffix
        test_file = self.dataset_path + self.dataset_test_filename + suffix
        print('saving sparse matrices as npz', end='    ')
        time_start = time.time()
        sparse.save_npz(train_file, matrix_train)
        sparse.save_npz(test_file, matrix_test)
        print('done in {} seconds'.format(time.time() - time_start))


if __name__ == '__main__':
    preprocess_test = PreProcessing()
    preprocess_test.split_data()

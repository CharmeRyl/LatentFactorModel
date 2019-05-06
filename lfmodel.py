from scipy import sparse
from scipy.sparse import linalg
from collections import defaultdict
import numpy as np
import json
import time
import datetime
import os


class LFModel:

    # directory paths
    dataset_path = 'dataset/'
    model_path = 'models/'

    # filenames
    train_filename = 'data_train.npz'
    test_filename = 'data_test.npz'

    # filepaths
    train_filepath = dataset_path + train_filename
    test_filepath = dataset_path + test_filename

    def __init__(self, k, epochs, learning_rate, lambda_r):
        # set parameters
        self.k = k
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_r = lambda_r
        self.model_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # init variables
        self.Q = None
        self.P = None
        # ensure the models path exists
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        # start initializing the model
        print('initializing model parameters', end='    ')
        time_start = time.time()
        # load matrices (coo format)
        self.train_mat = sparse.load_npz(self.train_filepath)
        self.test_mat = sparse.load_npz(self.test_filepath)
        # normalize the data
        self.user_avg = self.__user_rating_average__(self.train_mat)
        self.train_mat = self.__normalize_matrix_col__(self.train_mat.tocsc(), self.user_avg)
        self.train_mat_csr = self.train_mat.tocsr()
        print('done in {} seconds'.format(time.time() - time_start))

    def __normalize_matrix_col__(self, matrix_csc, col_avg):
        indptr = matrix_csc.indptr
        for j in range(matrix_csc.shape[1]):
            matrix_csc.data[indptr[j]:indptr[j + 1]] = matrix_csc.data[indptr[j]:indptr[j + 1]] - col_avg[j]
        return matrix_csc.tocoo()

    def __user_rating_average__(self, matrix_csc):
        global_average = matrix_csc.data.mean()
        user_num = matrix_csc.shape[1]
        user_sums = matrix_csc.sum(axis=0).getA1()
        user_cnts = matrix_csc.getnnz(axis=0)
        user_average = np.zeros(user_num)
        for j in range(user_num):
            user_average[j] = user_sums[j]/user_cnts[j] if user_sums[j] > 0 else global_average
        return user_average

    def __svd__(self, matrix_csc, k):
        u, s, vt = sparse.linalg.svds(matrix_csc, k=k)
        q = u
        p = np.diag(s).dot(vt)
        return q, p

    def __predict__(self, movie, user):
        return self.Q[movie, :].dot(self.P[:, user])

    def __predict_all_array__(self, matrix_coo, user_avg=None):
        if user_avg is None:
            user_avg = defaultdict(int)
        ratings_num = matrix_coo.getnnz()
        movies = matrix_coo.row
        users = matrix_coo.col
        predict_array = np.zeros(ratings_num)
        for i in range(ratings_num):
            predict_array[i] = self.__predict__(movies[i], users[i]) + user_avg[users[i]]
        return predict_array

    def __error__(self, movie, user, rating):
        predict_rating = self.__predict__(movie, user)
        return rating - predict_rating

    def __calc_rmse__(self, matrix_coo, user_avg=None):
        predict_array = self.__predict_all_array__(matrix_coo, user_avg=user_avg)
        mse = np.square(predict_array - matrix_coo.data).mean(axis=0)
        return np.sqrt(mse)

    def __shuffle__(self, matrix_coo):
        # get (row, col, data) pairs
        row, col, data = matrix_coo.row, matrix_coo.col, matrix_coo.data
        # shuffle the indices
        indices = np.arange(matrix_coo.getnnz())
        np.random.shuffle(indices)
        return row[indices], col[indices], data[indices]

    def get_rmse(self):
        print('calculating rmse', end='    ')
        time_start = time.time()
        train_rmse = self.__calc_rmse__(self.train_mat)
        test_rmse = self.__calc_rmse__(self.test_mat, user_avg=self.user_avg)
        print('done in {} seconds'.format(time.time() - time_start))
        return train_rmse, test_rmse

    def train(self):
        if self.Q is None or self.P is None:
            self.Q, self.P = self.__svd__(self.train_mat.tocsc(), self.k)
        print('rmse: {}'.format(self.get_rmse()))
        print('start training')
        time_train_start = time.time()
        for epoch in range(self.epochs):
            time_epoch_start = time.time()
            feed_count = 0
            total_count = self.train_mat.getnnz()
            row, col, data = self.__shuffle__(self.train_mat)
            for movie, user, rating in zip(row, col, data):
                grad_q = (2 * self.__error__(movie, user, rating) * self.P[:, user]).T - 2 * self.lambda_r * self.Q[movie, :]
                self.Q[movie, :] = self.Q[movie, :] + self.learning_rate * grad_q
                grad_p = (2 * self.__error__(movie, user, rating) * self.Q[movie, :]).T - 2 * self.lambda_r * self.P[:, user]
                self.P[:, user] = self.P[:, user] + self.learning_rate * grad_p
                feed_count += 1
                if feed_count % 100000 == 0:
                    print('stage {}/{} of epoch {} achieved in {} seconds'
                          .format(feed_count, total_count, epoch, time.time() - time_epoch_start))
            print('epoch {} done in {} seconds'.format(epoch, time.time() - time_epoch_start))
            self.save(epoch + 1)
        print('training with {} epochs done in {} seconds'.format(self.epochs, time.time() - time_train_start))
        self.save(self.epochs)

    def save(self, epoch):
        model_filepath = self.model_path + self.model_name + '.model'
        param_filepath = self.model_path + self.model_name + '.json'
        np.save(model_filepath, [self.Q, self.P])
        params = dict(
            model_name=self.model_name,
            latent_dimen=self.k,
            learning_rate=self.learning_rate,
            lambda_regu=self.lambda_r,
            curr_epochs=epoch,
            rmse=self.get_rmse()
        )
        with open(param_filepath, 'w') as param_file:
            json.dump(params, param_file)
        print('rmse: {}'.format(params['rmse']))
        print('model saved')

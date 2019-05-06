# def train_tf(self):
#     if self.Q is None or self.P is None:
#         self.Q, self.P = self.__svd__(self.train_mat.tocsc(), self.k)
#
#     q = tf.Variable(self.Q, dtype=tf.float32, name="Q")
#     p = tf.Variable(self.P, dtype=tf.float32, name="P")
#
#     x = tf.placeholder(tf.int32, [None, 2], name=None)
#     y = tf.placeholder(tf.float32, [None, 1], name=None)
#
#     row = tf.gather(q, x[0][0], axis=0)
#     col = tf.gather(p, x[0][1], axis=1)
#     y_pred = tf.reduce_sum(tf.multiply(row, col), name="Prediction")
#     mse = tf.reduce_mean(tf.square(y_pred - y), name="MSE")
#     grad = tf.gradients(mse, [row, col])
#
#     indices_q = tf.py_func(lambda idx: np.array([[int(idx[0]), i] for i in range(10)]), [x[0]], tf.int32)
#     updates_q = row - self.learning_rate * grad[0]
#     train_op_q = tf.scatter_nd_update(q, indices_q, updates_q)
#
#     indices_p = tf.py_func(lambda idx: np.array([[i, int(idx[1])] for i in range(10)]), [x[0]], tf.int32)
#     updates_p = col - self.learning_rate * grad[1]
#     train_op_p = tf.scatter_nd_update(p, indices_p, updates_p)
#
#     train_x = np.array(list(zip(self.train_mat.row, self.train_mat.col)))
#     train_y = np.expand_dims(np.array(self.train_mat.data), axis=1)
#
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         sess.run(init)
#         for epoch in range(self.epochs):
#             start = time.time()
#             for j in range(len(train_x)):
#                 if j % 100000 == 0:
#                     print("Current count {}".format(j))
#                     end = time.time()
#                     print("Time taken {}".format(end - start))
#                     start = end
#                 sess.run(train_op_q, feed_dict={x: [train_x[j]], y: [train_y[j]]})
#                 sess.run(train_op_p, feed_dict={x: [train_x[j]], y: [train_y[j]]})
#             print("Epoch {}: mse={}".format(epoch, mse.eval()))

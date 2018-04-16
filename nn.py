import tensorflow as tf
import numpy as np


class Graph:

	def __init__(self):

		self.inputs = tf.placeholder(tf.float32, shape=[None, 2])
		self.labels = tf.placeholder(tf.float32, shape=[None, 1])

		self.layer1 = tf.layers.dense(self.inputs, 16, activation=tf.nn.sigmoid)
		self.outputs = tf.layers.dense(self.layer1, 1, activation=tf.nn.sigmoid)

		# 最低にしたい対象
		self.cost = tf.losses.log_loss(self.outputs, self.labels)
		# 変数を変えて対象を最低にするオプティマイザ
		self.optim = tf.train.AdamOptimizer(0.01).minimize(self.cost)

		self.session = None


	def start(self): # セッションを始める
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer()) # 変数の初期化子
		return self.session

	def train(self, inputs, labels): # 学習する
		self.session.run(self.optim, feed_dict={self.inputs: inputs, self.labels: labels})

	def getCost(self, inputs, labels): # 学習する
		return self.session.run(self.cost, feed_dict={self.inputs: inputs, self.labels: labels})

	def prediction(self, inputs):
		return self.session.run([self.outputs], feed_dict={self.inputs: inputs})

	def write(self): # 表示のためにログを記録する
		self.writer = tf.summary.FileWriter("logs", self.session.graph)

import tensorflow as tf
import numpy as np


class Graph:

	def __init__(self):

		# 実際の状態
		self.labels = tf.placeholder(tf.float32, shape=[None, 1])

		# 入力層
		self.inputs = tf.placeholder(tf.float32, shape=[None, 2])

		# 隠れそう
		self.layer1 = tf.layers.dense(self.inputs, 16, activation=tf.nn.relu)
		self.layer2 = tf.layers.dense(self.layer1, 16, activation=tf.nn.relu)

		# 出力層/予測する状態
		self.outputs = tf.layers.dense(self.layer2, 1, activation=tf.nn.sigmoid)

		# エラー率
		self.cost = tf.losses.log_loss(self.outputs, self.labels)
		# 変数を変えてエラー率を最低にするオプティマイザ
		self.optim = tf.train.AdamOptimizer(0.005).minimize(self.cost)

		self.session = None


	def start(self): # セッションを始める
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer()) # 変数の初期化子
		return self.session

	def train(self, inputs, labels): # 学習する
		self.session.run(self.optim, feed_dict={self.inputs: inputs, self.labels: labels})

	def getCost(self, inputs, labels): # 現在のエラー率を計算する
		return self.session.run(self.cost, feed_dict={self.inputs: inputs, self.labels: labels})

	def prediction(self, inputs): # 状態を予測する
		return self.session.run([self.outputs], feed_dict={self.inputs: inputs})

	def write(self): # 表示のためにログを記録する
		self.writer = tf.summary.FileWriter("logs", self.session.graph)

import tensorflow as tf
import numpy as np


class Graph:

    def __init__(self): # グラフを準備する

        self.value = tf.placeholder(tf.float32, shape=[]) # 数値のパラメーターのノード
        self.matrix = tf.Variable(tf.zeros(dtype=tf.float32, shape=[5,5])) # 行列の変数
        self.add = tf.add(self.matrix, self.value) # 追加のノード
        self.assign = self.matrix.assign(self.add) # 行列の変数を設定する
        self.session = None


    def start(self): # セッションを始める

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer()) # 変数の初期化子
        return self.session

    def run(self, value): # 追加のノードを実行する
        r = self.session.run(self.assign, feed_dict={self.value: value})
        return r

    def write(self): # 表示のためにログを記録する
        self.writer = tf.summary.FileWriter("logs", self.session.graph)

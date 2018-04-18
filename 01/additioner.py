import tensorflow as tf
import numpy as np


class Graph:

    def __init__(self): # グラフを準備する

        self.matrix = tf.placeholder(tf.float32, shape=[5, 5]) # 行列のパラメーターのノード
        self.one = tf.constant(1.0, dtype=tf.float32, shape=[]) # 定数のノード
        self.add = tf.add(self.matrix, self.one) # 追加のノード
        self.session = None


    def start(self): # セッションを始める

        self.session = tf.Session()
        return self.session

    def run(self, matrix): # 追加のノードを実行する
        return self.session.run(self.add, feed_dict={self.matrix: matrix})

    def write(self): # 表示のためにログを記録する
        self.writer = tf.summary.FileWriter("logs", self.session.graph)

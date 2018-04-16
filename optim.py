import tensorflow as tf
import numpy as np


class Graph:

    def __init__(self): # ' A + B * x + C * x^2 ' のオプティマイザ

        self.A = tf.placeholder(tf.float32, shape=[]) # A
        self.B = tf.placeholder(tf.float32, shape=[]) # B
        self.C = tf.placeholder(tf.float32, shape=[]) # C
        self.x = tf.Variable(tf.random_uniform(dtype=tf.float32, shape=[])) # x
        self.Bx = tf.multiply(self.B, self.x) # B * x
        self.Cx2 = tf.multiply(self.C, tf.square(self.x)) # C * (x^2)

        # 最低にしたい対象
        self.cost = tf.add(self.A, tf.add(self.Bx, self.Cx2)) # A + B * x + C * x^2 
        # 変数を変えて対象を最低にするオプティマイザ
        self.optim = tf.train.GradientDescentOptimizer(0.1).minimize(self.cost)

        self.session = None


    def start(self): # セッションを始める

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer()) # 変数の初期化子
        return self.session

    def train(self, a, b, c): # 学習する
        self.session.run(self.optim, feed_dict={self.A: a, self.B: b, self.C: c})

    def result(self, a, b, c): # 現在の結果
        return self.session.run([self.cost, self.x], feed_dict={self.A: a, self.B: b, self.C: c})

    def write(self): # 表示のためにログを記録する
        self.writer = tf.summary.FileWriter("logs", self.session.graph)

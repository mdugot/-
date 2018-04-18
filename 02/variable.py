import tensorflow as tf
import numpy as np


class Graph:

    def __init__(self): # グラフを準備する

        self.value = tf.placeholder(tf.float32, shape=[]) # 数値のパラメーターのノード
        self.variable = tf.Variable(tf.zeros(dtype=tf.float32, shape=[])) # 変数
        self.add = tf.add(self.variable, self.value) # 追加のノード
        self.assign = self.variable.assign(self.add) # 変数を設定する
        self.session = None

        # Tensorboardの表示のため
        tf.summary.scalar("variable", self.assign)
        self.summary = tf.summary.merge_all()
        self.step = 0


    def start(self): # セッションを始める

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer()) # 変数の初期化子

        # Tensorboardの表示のため
        self.writer = tf.summary.FileWriter("logs", self.session.graph)
        return self.session

    def run(self, value): # 追加のノードを実行する
        r, summary = self.session.run([self.assign, self.summary], feed_dict={self.value: value})

        # Tensorboardの表示のため
        self.writer.add_summary(summary, self.step)
        self.writer.flush()
        self.step += 1

        return r

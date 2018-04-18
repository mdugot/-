import tensorflow as tf

class Graph:

    def __init__(self) :

        self.pixels = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
        self.labels = tf.placeholder(dtype = tf.int32, shape = [None])
    
        self.flattten = tf.contrib.layers.flatten(self.pixels)
    
        self.outputs = tf.layers.dense(self.flattten, 62, tf.nn.relu)
    
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels, logits = self.outputs))
    
        self.training = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
        self.prediction = tf.argmax(self.outputs, 1)
    
    def start(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        return self.session
    
    def train(self, pixels, labels):
        return  self.session.run([self.training, self.cost], feed_dict={self.pixels: pixels, self.labels: labels})
    
    
    def predict(self, pixels):
        return  self.session.run(self.prediction, feed_dict={self.pixels: pixels})

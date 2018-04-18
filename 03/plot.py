import matplotlib.pyplot as plt
import numpy as np
from optim import Graph

class Plot:

    def __init__(self, a = 2000, b = -200, c = 2):
        self.G = Graph()
        self.G.start()
        self.a = a
        self.b = b
        self.c = c
        plt.ion()

    def function(self, x):
        return self.a + self.b * x + self.c * (x ** 2)

    def train(self, learningRate):
        return self.G.train(learningRate, self.a, self.b, self.c)
    
    def draw(self):

        plt.clf()
        self.ax = plt.axes()
        x = np.linspace(-100, 200, 1000)
        self.ax.plot(x, self.function(x))
        Y,X = self.G.result(self.a, self.b, self.c)
        plt.scatter(X,Y,c='red')
        plt.xticks([-100,0,100,X], ["-100","0","100","X"])
        plt.show()

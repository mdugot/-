import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

class Plot:

	def testPrediction(self, x, y):
		return self.G.prediction([[x,y]])[0][0,0]
	
	def contour(self):
		r = np.arange(-1, 1, 0.1)
		l = len(r)
		c_x, c_y = np.meshgrid(r, r)
		c_z = np.zeros([l,l])
		for i,x in enumerate(r):
			for j,y in enumerate(r):
				c_z[i,j] = self.testPrediction(x, y)
		return {"x":c_x, "y":c_y, "z":c_z}
	
	def train(self, iterations):
		for i in range(iterations):
			self.G.train(self.data[:,0:2], np.transpose([self.data[:,2]]))
			if i % 100 == 0:
				print(str(self.G.getCost(self.data[:,0:2], np.transpose([self.data[:,2]]))))
				self.plotTraining()
		plt.show()

	def __init__(self):
		plt.ion()
		self.data = np.genfromtxt("data.txt", delimiter=",")
		self.trueData = self.data[self.data[:,2] == 1][:,0:2]
		self.falseData = self.data[self.data[:,2] == 0][:,0:2]

		self.G = nn.Graph()
		self.G.start()



	def plotTraining(self):

		#self.train(1000)
		self.c = self.contour()
		plt.contour(self.c["x"], self.c["y"], self.c["z"], colors='blue', levels=[0.5], alpha=0.3)
		plt.scatter(self.falseData[:,0], self.falseData[:,1], c='red', marker='+')
		plt.scatter(self.trueData[:,0], self.trueData[:,1], c='green', marker='o', edgecolor='black')
		

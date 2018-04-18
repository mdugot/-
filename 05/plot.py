import os
import nn
import numpy as np
import skimage
import matplotlib.pyplot as plt 
from skimage import data
from skimage import transform
from skimage.color import rgb2gray
from tqdm import tqdm
import random

class Plot:

    def __init__(self):

        self.G = nn.Graph()
        self.G.start()

        print("prepare training images")
        self.images_train,self.images28_train,self.labels_train = self.loadImages("./TrafficSigns/Training")
        print("prepare testing images")
        self.images_test,self.images28_test,self.labels_test = self.loadImages("./TrafficSigns/Testing")

    def loadImages(self, data_directory):

        directories = [d for d in os.listdir(data_directory) 
            if os.path.isdir(os.path.join(data_directory, d))]
        labels = []
        images = []
        print("load images")
        for d in tqdm(directories):
            label_directory = os.path.join(data_directory, d)
            file_names = [os.path.join(label_directory, f) 
                for f in os.listdir(label_directory) 
                if f.endswith(".ppm")]
            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))
        print("resize all images to the same dimensions")
        images28 =  [transform.resize(image, (28, 28)) for image in images]
        print("convert colors to gray level")
        images28 =  rgb2gray(np.array(images28))
        return images,images28,labels
    
    def plotSample(self):
        unique_labels = set(self.labels_train)
        i = 1
        plt.figure(figsize=(10, 10))
        for l in  unique_labels:
            plt.subplot(8,8,i)
            plt.axis("off")
            plt.title("{0}, ({1})".format(l, self.labels_train.count(l)))
            plt.imshow(self.images_train[self.labels_train.index(l)], cmap="gray")
            i += 1
        plt.tight_layout()
        plt.show()
    
    
    def checkSample(self):
    
        predictions = self.G.predict(self.images28_test)
        plt.figure(figsize=(10, 10))
    
        correctPredictions = 0
        for i in range(len(predictions)):
            if predictions[i] == self.labels_test[i]:
                correctPredictions += 1
        plt.subplot(9,1, 1)
        plt.axis("off")
        plt.text(0.5, 0.5, "{0} / {1}".format(correctPredictions, len(predictions)), fontsize=16, color="black")
    
        ri = random.sample(range(len(self.labels_test)), 64)
        i = 1
        for j in ri:
            plt.subplot(9,8,i+8)
            plt.axis("off")
            if self.labels_test[j] == predictions[j]:
                plt.text(0, -2, "{0} / {1}".format(self.labels_test[j], predictions[j]), fontsize=10, color="green")
            else:
                plt.text(0, -2, "{0} / {1}".format(self.labels_test[j], predictions[j]), fontsize=10, color="red")
            plt.imshow(self.images28_test[j], cmap="gray")
            i += 1
        plt.tight_layout()
        plt.show()
    
    def train(self, iterations):
        for i in range(iterations):
            _,loss_val = self.G.train(self.images28_train, self.labels_train)
            if i % 10 == 0:
                print("loss:", loss_val)

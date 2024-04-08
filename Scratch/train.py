#necessary libraries
import os
import sys
import math
import random
import numpy as np
from natsort import natsorted
from PIL import Image
# import pandas as pd
# from matplotlib import pyplot as plt

#make command line arguents (cla) that allow for something
#like taking the path to TestingImages and TestingLabels
#it then automatically goes through this entire folder and mayb
#consider each batch size 10 and does that
#also allows u to specify an epoch in the

#10 neurons in the output layer but i want classification
#only to giv me one of these 10

class FNN:
    def __init__(self, iLayerNeurons, hLayers, hLayerNeurons, oLayerNeurons,
                 miniBatchSize, epoch):
        self.iLayerNeurons = iLayerNeurons
        self.hLayers = hLayers
        self.hLayerNeurons = hLayerNeurons
        self.oLayerNeurons = oLayerNeurons
        self.miniBatchSize = miniBatchSize
        self.epoch = epoch
        self.weights = []
        self.biases = []
        self.trainingImages = []
        self.trainingLabels = []
        self.miniBatchImages = []
        self.miniBatchLabels = []

    def prepareTD(self, imagePath, labelPath):
        allImages = natsorted(os.listdir(imagePath))
        self.dLength = len(allImages)
        for images in allImages:
            path = os.path.join(imagePath, images)
            image = Image.open(path)
            # #confirms that the image is black and white
            if image.mode != "L":
                image = image.convert("L")
                print("The image was converted to grayscale.")
            # #confirms that the number of neurons in the input layer
            # #is the total amount of pixels per image
            # #assuming its a square
            if image.size[0] * image.size[1] != (self.iLayerNeurons):
                newSize = math.sqrt(self.iLayerNeurons)
                image = image.resize((newSize, newSize))
                print(f"The image was resized to {image.size[0]} by {image.size[1]}.")
            image_array = np.array(image)
            # #makes it 1-D
            vector = image_array.flatten()
            # #makes it a physically vertical vector
            # vector = vector.reshape((self.iLayerNeurons, 1))
            # #transposes it but it is still a 1-D vector (doesn't change)
            # # vector = vector.T
            self.trainingImages.append(vector.tolist())
            print(f"{images} with {len(vector.tolist())} elements was added to self.trainingImages")
        print(f"The trainingImages dataset is complete with {len(self.trainingImages)} entries.")

        allLabels = natsorted(os.listdir(labelPath))
        for labels in allLabels:
            path = os.path.join(labelPath, labels)
            with open(path, "r") as l:
                label = l.read()
                self.trainingLabels.append(label)
                print(f"{labels} contains a(n) {label} and was added to self.trainingLabels")
        print(f"The trainingLabels dataset is complete with {len(self.trainingLabels)} entries.")
        return

    def shuffleDataset(self, trainingImages, trainingLabels):
        pairedDataset = list(zip(trainingImages, trainingLabels))
        np.random.shuffle(pairedDataset)
        self.trainingImages, self.trainingLabels = zip(*pairedDataset)
        self.trainingImages = list(self.trainingImages)
        self.trainingLabels = list(self.trainingLabels)
        print("The dataset was shuffled.")
        return

    def prepareMiniBatch(self, miniBatchSize, epoch):
        for i in range(epoch):
            list1 = []
            for j in range(0, len(self.trainingImages), miniBatchSize):
                list1.append(self.trainingImages[j:miniBatchSize + j])
            self.miniBatchImages.append(list1)
            list2 = []
            for k in range(0, len(self.trainingLabels), miniBatchSize):
                list2.append(self.trainingLabels[k:miniBatchSize + k])
            self.miniBatchLabels.append(list2)
        print("self.miniBatchImages is complete.")
        print("self.miniBatchLabels is complete.")
        return

    def prepareInputWeights(self):
        list1 = []
        for i in range(self.iLayerNeurons):
            list2 = []
            for j in range(self.hLayerNeurons):
                list2.append(random.random() - 0.5)
            list1.append(list2)
        self.weights.append(list1)
        print("Input layer weights complete.")
        return

    def prepareHiddenWeights(self):
        for i in range(self.hLayers - 1):
            list1 = []
            for j in range(self.hLayerNeurons):
                list2 = []
                for k in range(self.hLayerNeurons):
                    list2.append(random.random() - 0.5)
                list1.append(list2)
            self.weights.append(list1)
            print(f"Hidden layer weights complete.")
        return

    def prepareOutputWeights(self):
        list1 = []
        for i in range(self.hLayerNeurons):
            list2 = []
            for j in range(self.oLayerNeurons):
                list2.append(random.random() - 0.5)
            list1.append(list2)
        self.weights.append(list1)
        print("Output layer weights complete.")
        return

    def prepareWeights(self):
        self.prepareInputWeights()
        self.prepareHiddenWeights()
        self.prepareOutputWeights()
        return

    def getWeights(self):
        return self.weights

    def totalWeights(self):
        try:
            with open("weights.txt", "r") as file:
                text = file.read().replace("[", "").replace("]", "").replace(",", "").split()
                print(f"{self.iLayerNeurons}x{self.hLayerNeurons} + {self.hLayerNeurons}x{self.hLayerNeurons} + {self.hLayerNeurons}x{self.oLayerNeurons} = {len(text)} weights")
        except:
            print("The weights.txt file does not exist.")

    def prepareInputBiases(self):
        pass

    def prepareHiddenBiases(self):
        for i in range(self.hLayers):
            list1 = []
            for j in range(self.hLayerNeurons):
                list1.append(random.random() - 0.5)
            self.biases.append(list1)
            print(f"Hidden {i + 1} biases complete.")
        return

    def prepareOutputBiases(self):
        list1 = []
        for i in range(self.oLayerNeurons):
            list1.append(random.random() - 0.5)
        self.biases.append(list1)
        print("Output layer biases complete.")
        return

    def prepareBiases(self):
        self.prepareInputBiases()
        self.prepareHiddenBiases()
        self.prepareOutputBiases()
        return

    def getBiases(self):
        return self.biases

    def totalBiases(self):
        try:
            with open("biases.txt", "r") as file:
                text = file.read().replace("[", "").replace("]", "").replace(",", "").split()
                print(f"{self.hLayerNeurons} + {self.hLayerNeurons} + {self.oLayerNeurons} = {len(text)} biases")
        except:
            print("The biases.txt file does not exist.")

    def getTrainingImages(self):
        return self.trainingImages

    def getTrainingLabels(self):
        return self.trainingLabels

    def printAll(self):
        print(f"Inputs: {self.iLayerNeurons}")
        print(f"Hidden Layers: {self.hLayers}")
        print(f"Neurons per Hidden Layer: {self.hLayerNeurons}")
        print(f"Different Classifications: {self.oLayerNeurons}")
        print(f"Length of each Mini-Batch: {self.miniBatchSize}")
        print(f"Epoch: {self.epoch}")
        return

    def getRandInputVector(self):
        return self.dLength

def resetTextFile(fName):
    open(fName, "w").close()
    print(f"{fName} was reset.")

def writeTextFile(fName, content):
    with open(fName, "w") as file:
        file.write(str(content))
    print(f"{fName} was written.")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit("Please specify the mini-batch size and epoch.")
    if sys.argv[1] == "default":
        miniBatchSize = int(sys.argv[2])
        epoch = int(sys.argv[3])
        defaultFNN = FNN(784, 2, 16, 10, miniBatchSize, epoch)

    defaultFNN.printAll()

    defaultFNN.prepareTD(r"C:\Users\NCallabresi\Documents\PythonProjects\mnistTesting\MNIST\TrainingImages",
                         r"C:\Users\NCallabresi\Documents\PythonProjects\mnistTesting\MNIST\TrainingLabels")

    defaultFNN.shuffleDataset(defaultFNN.trainingImages, defaultFNN.trainingLabels)

    defaultFNN.prepareMiniBatch(defaultFNN.miniBatchSize, defaultFNN.epoch)
    resetTextFile("miniBatchImages.txt")
    writeTextFile("miniBatchImages.txt", defaultFNN.miniBatchImages)
    print("miniBatchImages:")
    print(len(defaultFNN.miniBatchImages))
    print(type(defaultFNN.miniBatchImages))
    print(len(defaultFNN.miniBatchImages[0]))
    print(type(defaultFNN.miniBatchImages[0]))
    print(len(defaultFNN.miniBatchImages[0][0]))
    print(type(defaultFNN.miniBatchImages[0][0]))
    resetTextFile("miniBatchLabels.txt")
    writeTextFile("miniBatchLabels.txt", defaultFNN.miniBatchLabels)
    print("miniBatchLabels:")
    print(len(defaultFNN.miniBatchLabels))
    print(type(defaultFNN.miniBatchLabels))
    print(len(defaultFNN.miniBatchLabels[0]))
    print(type(defaultFNN.miniBatchLabels[0]))
    print(len(defaultFNN.miniBatchLabels[0][0]))
    print(type(defaultFNN.miniBatchLabels[0][0]))

    defaultFNN.prepareWeights()
    resetTextFile("weights.txt")
    writeTextFile("weights.txt", defaultFNN.getWeights())
    print("Weights:")
    print(len(defaultFNN.getWeights()))
    print(type(defaultFNN.getWeights()))
    print(len(defaultFNN.getWeights()[0]))
    print(len(defaultFNN.getWeights()[1]))
    print(len(defaultFNN.getWeights()[2]))
    print(type(defaultFNN.getWeights()[0]))
    print(len(defaultFNN.getWeights()[0][0]))
    print(len(defaultFNN.getWeights()[1][0]))
    print(len(defaultFNN.getWeights()[2][0]))
    print(type(defaultFNN.getWeights()[0][0]))
    defaultFNN.totalWeights()

    defaultFNN.prepareBiases()
    resetTextFile("biases.txt")
    writeTextFile("biases.txt", defaultFNN.getBiases())
    # print("Biases:")
    # print(len(defaultFNN.getBiases()))
    # print(type(defaultFNN.getBiases()))
    # print(len(defaultFNN.getBiases()[0]))
    # print(type(defaultFNN.getBiases()[0]))
    defaultFNN.totalBiases()

    # print(defaultFNN.getInputVector())
    # print(defaultFNN.getRandInputVector())
    # print(len(defaultFNN.trainingLabels))

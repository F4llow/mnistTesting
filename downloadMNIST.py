# import torchvision
#
# trainset = torchvision.datasets.MNIST(root = "./",
#                                       train = True,
#                                       download = True)

################################################################################

# import os
# import gzip
# import numpy as np
#
# root = "./"
#
# train_images_path = os.path.join(root, 'MNIST', 'raw', 'train-images-idx3-ubyte')
# train_labels_path = os.path.join(root, 'MNIST', 'raw', 'train-labels-idx1-ubyte')
#
# def read_images_from_file(path):
#     with open(path, "rb") as file:
#         data = np.frombuffer(file.read(), dtype = np.uint8, offset = 16)
#     return data.reshape(-1, 28, 28)
#
# def read_labels_from_file(path):
#     with open(path, "rb") as file:
#         data = np.frombuffer(file.read(), dtype = np.uint8, offset = 8)
#     return data
#
# train_images = read_images_from_file(train_images_path)
# train_labels = read_labels_from_file(train_labels_path)
#
# print("Train images shape:", train_images.shape)
# print("Train labels shape:", train_labels.shape)

################################################################################

# import torchvision.datasets as tv
# import cv2
# import numpy as np
#
# trainset = tv.MNIST(root = "./", train = True, download = False)
#
# for i in range(len(trainset)):
#     image, label = trainset[i]
#     image = np.array(image)
#     image = 255 - image
#
#     filename1 = fr"C:\Users\NCallabresi\Documents\PythonProjects\mnistTesting\newMNIST\TrainingImages\train_image_{i + 1}.jpg"
#     cv2.imwrite(filename1, image)
#     print(f"Picture number {i + 1} was downloaded.")
#
#     filename2 = fr"C:\Users\NCallabresi\Documents\PythonProjects\mnistTesting\newMNIST\TrainingLabels\train_label{i + 1}.txt"
#     with open(filename2, "w") as file:
#         file.write(str(label))
#     print(f"Label number {i + 1} was downloaded.")

################################################################################

# import torchvision.datasets as tv
# import cv2
# import numpy as np
#
# testset = tv.MNIST(root = "./", train = False, download = False)
#
# for i in range(len(testset)):
#     image, label = testset[i]
#     image = np.array(image)
#     image = 255 - image
#
#     filename1 = fr"C:\Users\NCallabresi\Documents\PythonProjects\mnistTesting\newMNIST\TestingImages\test_image_{i + 1}.jpg"
#     cv2.imwrite(filename1, image)
#     print(f"Picture number {i + 1} was downloaded.")
#
#     filename2 = fr"C:\Users\NCallabresi\Documents\PythonProjects\mnistTesting\newMNIST\TestingLabels\test_label{i + 1}.txt"
#     with open(filename2, "w") as file:
#         file.write(str(label))
#     print(f"Label number {i + 1} was downloaded.")

################################################################################

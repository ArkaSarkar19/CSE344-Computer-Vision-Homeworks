import  numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Blue_Winged_Warbler_0078_161889.jpg")
img = cv2.resize(img, (240, 240)) 
print(img.shape)
plt.imshow(img)
plt.show()

def calculateSppVector(img):
    parts = [4,9,16,25]
    vector = []
    for p in parts:
        size = int(p**0.5)
        n = int(img.shape[0]//size)
        for i in range(size):
            for j in range(size):
                curr = img[i*n: (i+1)*n,j*n: (j+1)*n, : ]
                vector.extend(calulateStats(curr))
    return vector
    

def calulateStats(matrix):
    vector = []
    for i in range(3):
        mean = np.mean(matrix[:,:,i])
        var = np.var(matrix[:,:,i])
        vector.append(mean)
        vector.append(var)
    return vector


vec = calculateSppVector(img)

print("Global Feature vector for the image is :", vec)
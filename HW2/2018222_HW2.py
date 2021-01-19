import cv2
import numpy as np 
import matplotlib.pyplot as plt
import copy


def threshold(image):

	R = image[:,:,0]
	G = image[:,:,1]
	B = image[:,:,2]


	# cv2.imshow('R',image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# cv2.imshow('G',G)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	# cv2.imshow('B',B)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	threshold_R = otsu(R)
	threshold_G = otsu(G)
	threshold_B = otsu(B)

	threshold = (threshold_R + threshold_G + threshold_B )>2

	cv2.imshow('thres_R',threshold_R*R)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imshow('thres_G',threshold_G*G)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


	cv2.imshow('thres_B',threshold_B*B)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	threshold = np.dstack((threshold,threshold,threshold))

	cv2.imshow('thres',1- threshold*np.ones(threshold.shape))
	cv2.waitKey(0)
	cv2.destroyAllWindows()



	return image*threshold



def otsu(img):

	min_cost = float('inf')
	threshold = 0

	for i in range(1,256):
		v0 = np.var(img[img < i], ddof = 1)
		w0 = len(img[img < i])
		v1 = np.var(img[img >= i], ddof = 1)
		w1 = len(img[img >= i])

		cost = w0*v0 + w1*v1
		if(cost < min_cost):
			min_cost = cost
			threshold = i

	return threshold




if __name__ == "__main__":

	# img = cv2.imread("dog-1020790_960_720.jpg", 2)
	# img.astype(int)



	# thres = otsu(img)


	# img[img<thres] = 777
		
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# image = cv2.imread("dog-1020790_960_720.jpg")
	# image.astype(int)
	# image[:,:,0][img==777] = 255
	# image[:,:,1][img==777] = 0
	# image[:,:,2][img==777] = 0



	# cv2.imshow('otsu',image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# print(otsu.shape)


	image = cv2.imread("dog-1020790_960_720.jpg")
	image.astype(int)

	i = threshold(image)

	cv2.imshow('otsu',i)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

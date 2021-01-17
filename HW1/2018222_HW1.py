import cv2
import numpy as np
from collections import deque

img = cv2.imread("Project1.png",cv2.IMREAD_GRAYSCALE)/255
img.astype(int)
c = 1
# for i in range(img.shape[0]):
# 	for j in range(img.shape[1]):
# 		# print(img[i][j], end = " ")
# 		if(img[i][j] == 1.0):
# 			c = c + 1
# 	# print()
# print(c)
print(img.shape)
rows, cols = img.shape[0], img.shape[1]
visited = np.zeros((rows,cols))

answer = np.zeros((rows,cols))

print(visited.shape)

for i in range(rows):
	for j in range(cols):

		if(img[i][j] == 0.0):
			visited[i,j] = 1
		elif(visited[i,j]):
			continue
		else:

			stack = deque()

			stack.append((i,j))

			while(len(stack)!=0):

				curr = stack.pop()

				if(visited[curr[0],curr[1]] == 0):

					visited[curr[0],curr[1]] = 1

					m,n = curr[0],curr[1]
					answer[m,n] = c

					list_ = []
					for x in range(m-1, m+2):
						for y in range(n-1, n+2):
							 if(x == m and y == n):
							 	continue
							 else:

							 	if(x <0 or x > rows -1):
							 		continue
							 	if(y <0 or y > cols -1):
							 		continue

							 	if(img[x,y] == 1.0):
							 		stack.append((x,y))
			c = c + 1


print("Number of connected components : ", np.amax(answer))







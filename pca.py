import os

path=r'E:\Shravya\pca'

#get the list of images
for img in os.listdir(path):
    print(img)
print('No of images',len(os.listdir(path)))

#Resizing all the images to 100 * 100 from PIL import Image import os, sys

path=r'E:\Shravya\pca' dirs = os.listdir( path )

def resize():
	for item in dirs:
		if os.path.isfile(os.path.join(path,item)): 
			print(item) 
			im = Image.open(os.path.join(path,item)) 
			f, e = os.path.splitext(os.path.join(path,item)) 
			imResize = im.resize((100,100), Image.ANTIALIAS) 
			imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

resize()

face_vector = []
import cv2
import numpy as np
for img in os.listdir(path):
	face_image = cv2.cvtColor(cv2.imread(os.path.join(path,img)), cv2.COLOR_RGB2GRAY)
    	face_image = face_image.reshape(10000,)
	face_vector.append(face_image)
face_vector = np.asarray(face_vector)
face_vector_transpose = face_vector.transpose()

face_vector_transpose.shape
print(face_vector[0].shape)

avg_face_vector = face_vector_transpose.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector_transpose.shape[0], 1)
normalized_face_vector = face_vector_transpose- avg_face_vector

import matplotlib.pyplot as plt
plt.imshow(face_vector[1].reshape(100,100), cmap='Greys')

plt.imshow(avg_face_vector.reshape(100,100), cmap='Greys')

plt.imshow(normalized_face_vector[:,1].reshape(100,100), cmap='Greys')

normalized_face_vector.shape

covariance_matrix = np.cov(np.transpose(normalized_face_vector))

cov1=np.dot(normalized_face_vector,np.transpose(normalized_face_vector))

covariance_matrix.shape

cov1.shape

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

eigvals, eigvecs = la.eig(covariance_matrix)
print(eigvals)

#eigvecs = sorted(eigvecs)
top_eig_vecs = eigvecs[0:16, :]

eigen_faces = top_eig_vecs.dot(normalized_face_vector.T)

weights = (normalized_face_vector.T).dot(eigen_faces.T)

eigen_faces.shape

plt.imshow(eigen_faces[0].reshape(100,100), cmap='Greys')

weights.shape

restructured_images=np.dot(weights,eigen_faces)

restructured_images.shape

plt.imshow(restructured_images[0].reshape(100,100), cmap='Greys')

eigvecs.shape
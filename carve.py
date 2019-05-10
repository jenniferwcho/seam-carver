# Jennifer Cho, Alex Cadigan, Tanush Samson
# COMP 398 Advanced Pictures & Sound 
# Seam Carving Project 

import numpy as np 
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image, ImageDraw
import cv2 

def open_image(path):
	"""
	Opens an image given
	"""
	image = Image.open(path)
	image.show()
	return image 

def calculate_map(image): 
	"""
	Computes energy value for each pixel. Gets "derivative" of given image and sums up abs vals 
	"""
	b, g, r = cv2.split(image)
	b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
	g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
	r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
	   
	return b_energy + g_energy + r_energy


def min_seam(image): 
	"""
	Finds a seam path from top to bottom with the least energy 
	"""
	row, col, channels = image.shape
	energy_map = calculate_map(image)

	M = energy_map.copy()
	backtrack = np.zeros_like(M, dtype = np.int)

	for r in range(1, row): 
		for c in range(0, col):
			if c == 0:
				index = np.argmin(M[r - 1, c:c + 2])
				backtrack[r, c] = index + c
				min_energy = M[r - 1, index + c]
			else: 
				index = np.argmin(M[r - 1, c - 1:c + 2])
				backtrack[r, c] = index + c - 1
				min_energy = M[r - 1, index + c - 1]

			M[r, c] += min_energy

	return M, backtrack 

# def draw_seam(image): 
# 	"""
# 	Draws seam with least energy path 
# 	"""
# 	row, col, channels = image.shape
# 	M, backtrack = min_seam(image)

# 	mask = np.ones((row, col), dtype = np.bool)
# 	c = np.argmin(M[-1])

# 	for r in reversed(range(row)):
# 		mask[r,c] = False
# 		c = backtrack[r,c]

# 	mask = np.logical_not(mask)
# 	image[...,0][mask] = 0 
# 	image[...,1][mask] = 0
# 	image[...,2][mask] = 255

# 	return image 


def carve(image):
	"""
	Deletes pixels from seam path with the least energy, returns new carved image
	"""
	row, col, channels = image.shape
	M, backtrack = min_seam(image)

	mask = np.ones((row,col), dtype = np.bool)
	c = np.argmin(M[-1])

	for r in reversed(range(row)):
		mask[r,c] = False
		c = backtrack[r,c]

	mask = np.stack([mask] * 3, axis=2)
	image = image[mask].reshape((r, c - 1, 3))

	return image





# def object_removal: 
# 	"""
# 	removes object in image
#   --let's try implementing this--
# 	"""






def main(): 
	image = cv2.imread("/Users/jenniferwcho/desktop/independentStudy/seamcarver/sea.png").astype(np.float64)
	calculate_map(image)

	original_image = open_image("/Users/jenniferwcho/desktop/independentStudy/seamcarver/sea.png")


main() 


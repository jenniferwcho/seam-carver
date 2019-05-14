# Jennifer Cho, Alex Cadigan, Tanush Samson
# COMP 398 Advanced Pictures & Sound 
# Seam Carving Project 

import numpy as np 
import matplotlib.pyplot as plt
import multiprocessing as mp
import cv2 
import time
import numba
from PIL import Image, ImageDraw
from numba import jit 




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

@numba.jit
def min_seam(image): 
	"""
	Finds a seam path from top to bottom with the least energy 
	"""
	row, col, channels = image.shape
	energy_map = calculate_map(image)

	M = energy_map.copy()
	backtrack = np.zeros_like(M, dtype = np.int)

	for i in range(1, row): 
		for j in range(0, col):
			if j == 0:
				index = np.argmin(M[i - 1, j:j + 2])
				backtrack[i, j] = index + j
				min_energy = M[i - 1, index + j]
			else: 
				index = np.argmin(M[i - 1, j - 1:j + 2])
				backtrack[i, j] = index + j - 1
				min_energy = M[i - 1, index + j - 1]

			M[i, j] += min_energy

	return M, backtrack 

@numba.jit
def carve(image):
	"""
	Deletes pixels from seam path with the least energy, returns new carved image
	"""
	row, col, channels = image.shape
	M, backtrack = min_seam(image)

	mask = np.ones((row,col), dtype = np.bool)
	j = np.argmin(M[-1])

	for i in reversed(range(row)):
		mask[i,j] = False
		j = backtrack[i,j]

	mask = np.stack([mask] * 3, axis=2)
	image = image[mask].reshape((row, col-1, 3))

	return image

def crop_by_col(image, col_scale):
	"""
	crops column
	"""
	row, col, channels = image.shape
	new_col = int(col_scale*col)

	for i in range(col - new_col): 
		image = carve(image)

	return image
		
def crop_by_row(image, row_scale): 
	"""
	crops row by rotating image and calling crop_by_col function
	"""
	image = np.rot90(image, 1, (0,1))
	image = crop_by_col(image, row_scale)
	image = np.rot90(image, 3, (0,1))

	return image

	

# def object_removal: 
# 	"""
# 	removes object in image
#   --let's try implementing this--
# 	"""



def main(): 

	image = cv2.imread("/Users/jenniferwcho/desktop/independentStudy/seamcarver/sea.png").astype(np.float64)
	scale = float(input("Please enter scaling value: "))
	print("Scaling Value: ", scale)

	start = time.time()
	#output_image = crop_by_col(image, scale) 
	output_image = crop_by_row(image, scale)
	end = time.time()

	cv2.imwrite("/Users/jenniferwcho/desktop/independentStudy/seamcarver/output_image/result.png", output_image)
	result_image = open_image("/Users/jenniferwcho/desktop/independentStudy/seamcarver/output_image/result.png")
	original_image = open_image("/Users/jenniferwcho/desktop/independentStudy/seamcarver/sea.png")


	execution_time = end - start
	print("Exection Time: ", execution_time)
	
	

	


main() 


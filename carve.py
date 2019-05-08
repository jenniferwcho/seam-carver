# Jennifer Cho, Alex Cadigan, Tanush Samson
# COMP 398 Advanced Pictures & Sound 
# Seam Carving Project 

import numpy as np 
import cv2 

def energy_map(image): 
	b, g, r = cv2.split(image)
	b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
	g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
	r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
	   
	return b_energy + g_energy + r_energy


image = cv2.imread("/Users/jenniferwcho/desktop/independentStudy/seamcarving/sea.png").astype(np.float64)
energy_map(image)


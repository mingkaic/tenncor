""" Utility settings for generating random stuff """

import random
import numpy as np

def randshape():
	return list(np.random.randint(1, high=9, size=randint()))

def randdouble():
	return random.random() * 37 - 17.4

def randint(min=1, max=9):
	return random.randint(min, max)

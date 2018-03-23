""" Shape propagation for ELEM class operations """

def ELEM(parent, makes):
	for make in makes:
		make(parent)

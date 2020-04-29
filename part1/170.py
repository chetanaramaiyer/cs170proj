import numpy as np
import random
#vertices
vertices = 5 #randomly gen

# def input(v):
# 	print(v) #prints vertices
# 	for i in range(v):
# 		for x in range(v):
# 			if i != x:
# 				isChosen = np.random.randint(0,8)
# 				#print("ischosen " + str(isChosen))
# 				if (isChosen == 0):
# 					w = round((random.random() * 100), 3)
# 					print(str(i) + " " + str(x) + " " + str(w))


#create empty list {}
b = [0]
index = 0

def input(v, start_index, a):
	if (start_index == v-1):
		return
	v1 = a[start_index]
	v2 = np.random.randint(0,v) #initial for before the while 
	backedge = []
	while v2 in a: #generating edges back to vertices that already exist in [a]
		weight = round((random.random() * 100), 3)
		if(a[start_index -1 ] != v2 and v1 != v2 and (v2 not in backedge)):
			print(str(v1) + " " + str(v2) + " " + str(weight))
			backedge += [v2]
		v2 = np.random.randint(0,v)
	
	a += [v2]
	weight = round((random.random() * 100), 3)
	print(str(v1) + " " + str(v2) + " " + str(weight))
	
	input(v, start_index + 1, a)

input(5, index, b)
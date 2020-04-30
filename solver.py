import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import random

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    vertices = list(G.nodes)
    edges = list(G.edges.data('weight'))
    res = KruskalMST(vertices, edges)

    T = nx.Graph()
    T.add_weighted_edges_from(res)
    avg_dist = average_pairwise_distance_fast(T)

    leaves = getLeaves(res)
    prunedEdges, prunedVertices = pruneLeavesDesc(leaves, res, avg_dist)
    T.remove_edges_from(prunedEdges)
    T.remove_nodes_from(prunedVertices)
    print(T.edges.data('weight'))
    print(T.nodes)
    
    return T

# Python program for Kruskal's algorithm to find 
# Minimum Spanning Tree of a given connected, 
# undirected and weighted graph 

def sortEdges(tup):
	#print(edges)  

    # getting length of list of tuples 
    lst = len(tup)  
    for i in range(0, lst):
          
        for j in range(0, lst-i-1):  
            if (tup[j][2] > tup[j + 1][2]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup

def KruskalMST(vertices, edges): 

	result = [] #This will store the resultant MST 

	i = 0 # An index variable, used for sorted edges 
	e = 0 # An index variable, used for result[] 

		# Step 1: Sort all the edges in non-decreasing 
			# order of their 
			# weight. If we are not allowed to change the 
			# given graph, we can create a copy of graph 
	sortedEdges = sortEdges(edges)
	#print(sortedEdges)

	parent = [] 
	rank = [] 

	# Create V subsets with single elements 
	for node in range(len(vertices)): 
		parent.append(node)
		rank.append(0)

	# Number of edges to be taken is equal to V-1 
	while e < len(vertices)-1 : 

		# Step 2: Pick the smallest edge and increment 
				# the index for next iteration 
		u,v,w = sortedEdges[i]
		i = i + 1
		x = find(parent, u) 
		y = find(parent ,v) 

		# If including this edge does't cause cycle, 
					# include it in result and increment the index 
					# of result for next edge 
		if x != y: 
			e = e + 1	
			result.append((u,v,w)) 
			union(parent, rank, x, y)			 
		# Else discard the edge 
	print(result)
	return result
	# print the contents of result[] to display the built MST 
	#print "Following are the edges in the constructed MST"
	# for u,v,weight in result: 
	# 	#print str(u) + " -- " + str(v) + " == " + str(weight) 
	# 	print ("%d -- %d == %d" % (u,v,weight)) 

def find(parent, i): 
    if parent[i] == i: 
        return i 
    return find(parent, parent[i]) 

# A function that does union of two sets of x and y 
# (uses union by rank) 
def union(parent, rank, x, y): 
    xroot = find(parent, x) 
    yroot = find(parent, y) 

    # Attach smaller rank tree under root of  
    # high rank tree (Union by Rank) 
    if rank[xroot] < rank[yroot]: 
        parent[xroot] = yroot 
    elif rank[xroot] > rank[yroot]: 
        parent[yroot] = xroot 

    # If ranks are same, then make one as root  
    # and increment its rank by one 
    else : 
        parent[yroot] = xroot 
        rank[xroot] += 1


def getLeaves(mst):
	'''
		input: mst graph
		output: 
	'''
	nodes = {}
	for u,v,w in mst:
		if u in nodes:
			nodes[u] = (-1,-1,-1)
		else:
			nodes[u] = (u,v,w)
		if v in nodes:
			nodes[v] = (-1,-1,-1)
		else:
			nodes[v] = (u,v,w)

	nodes = {x:y for x,y in nodes.items() if y!=(-1,-1,-1)}
	print(nodes)
	return nodes

def pruneLeavesDesc(l, res, currentAvg):
	#VERY naive solution bc we repeat avg_pairwise_distance method on EVERY call
	leaves = l.values()
	#print(leaves)
	removedLeaves = []
	temp = res.copy()
	verticesRemoved = {}
	#start with the largest edge weight of leaves first
	for elem in reverse(leaves):
		temp.remove(elem)
		print(temp)
		#create new graph to recalculate avg pairwise distance w/o elem
		Tr = nx.Graph()
		Tr.add_weighted_edges_from(temp)
		new_avg = average_pairwise_distance_fast(Tr)
		#if better avg obtained w/o elem: remove it and update avg
		#else, add it back and move on to next leaf
		if new_avg <= currentAvg:
			removedLeaves.append(elem)
			verticesRemoved.add([elem[0], elem[1]]) #add vertices to this set
			currentAvg = new_avg
			print("removed:"+ str(new_avg))
		else:
			temp.add(elem)
			print("kept:"+ str(new_avg))

	return (removedLeaves, verticesRemoved)

def pruneLeavesRando(l, currentAvg, res):
	#randomize the order and see if we get different solutions?
	leaves = sortEdges(l)
	print(leaves)
	removedLeaves = []
	temp = res.copy()
	bestAvg = currentAvg
	bestLeaves = []
	#start with the largest edge weight of leaves first
	for i in range(0, 15):
		leaves = random.shuffle(leaves)
		for elem in leaves:
			temp.remove(elem)
			print(temp)
			#create new graph to recalculate avg pairwise distance w/o elem
			Tr = nx.Graph()
			Tr.add_weighted_edges_from(temp)
			new_avg = average_pairwise_distance(Tr)
			#if better avg obtained w/o elem: remove it and update avg
			#else, add it back and move on to next leaf
			if new_avg <= currentAvg:
				removedLeaves.append(elem)
				currentAvg = new_avg
				print("removed:"+ str(new_avg))
			else:
				temp.add(elem)
				print("kept:"+ str(new_avg))
		if new_avg <= bestAvg:
			bestLeaves = removedLeaves.copy()
			bestAvg = new_avg
			print(new_avg + "    " + "iteration " +  i)
		removedLeaves = []


	return bestLeaves



	

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'out/test.out')

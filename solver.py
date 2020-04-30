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

    vertices = list(G.nodes)
    edges = list(G.edges.data('weight'))
    res = KruskalMST(vertices, edges) #creates mst

    T = nx.Graph()
    T.add_weighted_edges_from(res)
    avg_dist = average_pairwise_distance_fast(T) #current min pairwise avg of mst

    leaves = getLeaves(res)
    prunedEdges, prunedVertices, avgRand = pruneLeavesRando(leaves, res, avg_dist)
    prunedEdgesD, prunedVerticesD, avgDesc = pruneLeavesDesc(leaves, res, avg_dist)
    #chooses better pairwise avg between descending and randomized leaves
    if avgDesc < avgRand:
    	T.remove_edges_from(prunedEdgesD)
    	T.remove_nodes_from(prunedVerticesD)
    else:
    	T.remove_edges_from(prunedEdges)
    	T.remove_nodes_from(prunedVertices)

    return T

def sortEdges(tup):
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
	'''
		input:
			vertices: vertices of our input graph G in a list
			edges: edges of G in a list
		output: a list of edges of our resultant MST
	'''
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
	#print(result)
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
		output: a dictionary of leaves
			   {v : (u,v,w), ...} where v is a leaf with its corresponding edge
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
	#print(nodes)
	return nodes

def pruneLeavesDesc(l, res, currentAvg):
	'''
		input: l = all leaves of mst
			   res = complete mst
			   avg = mst's current avg 
		output: (best edges, best vertices) of leaves to be removed
				from mst to minimize avg
				runs in order from largest edge to lowest edge
	'''
	leaves = l
	removedLeaves = []
	temp = res.copy()
	verticesRemoved = []

	#start with the largest edge weight of leaves first
	def by_value(item):
		return item[1]

	for vertex, elem in sorted(leaves.items(), key=by_value, reverse=True):
		i = temp.index(elem) #index of curr elem; needed so we can add back to temp in right order
		temp.remove(elem)
		#print(temp)
		#create new graph to recalculate avg pairwise distance w/o elem
		Tr = nx.Graph()
		Tr.add_weighted_edges_from(temp)
		new_avg = average_pairwise_distance_fast(Tr)
		#if better avg obtained w/o elem: remove it and update avg
		#else, add it back and move on to next leaf
		if new_avg <= currentAvg:
			removedLeaves.append(elem)
			verticesRemoved.append(vertex) #add vertices to this set
			currentAvg = new_avg
		else:
			temp.insert(i, elem)
	print("descAvg:" + str(currentAvg))
	return (removedLeaves, verticesRemoved, currentAvg)

def pruneLeavesRando(l, res, avg):
	'''
		input: l = all leaves of mst
			   res = complete mst
			   avg = mst's current avg 
		output: (best edges, best vertices) of leaves to be removed
				from mst to minimize avg
	'''
	removedLeaves = []
	temp = res.copy()
	currentAvg = avg
	bestAvg = currentAvg
	bestLeaves = []
	verticesRemoved = []
	bestVerticesRemoved = []
	keys = list(l.keys())
	def by_value(item):
		return item[1]
	#start with the largest edge weight of leaves first
	for i in range(0, 15):
		#shuffle dictionary
		random.shuffle(keys)
		leaves = dict()
		for key in keys:
  			leaves.update({key:l[key]})

		for vertex, elem in leaves.items():
			ind = temp.index(elem)
			temp.remove(elem)

			#create new graph to recalculate avg pairwise distance w/o elem
			Tr = nx.Graph()
			Tr.add_weighted_edges_from(temp)
			new_avg = average_pairwise_distance_fast(Tr)
			#if better avg obtained w/o elem: remove it and update avg
			#else, add it back and move on to next leaf
			if new_avg <= currentAvg:
				removedLeaves.append(elem)
				verticesRemoved.append(vertex) #add vertices to this set
				currentAvg = new_avg
			else:
				temp.insert(i, elem)
		if new_avg <= bestAvg:
			bestLeaves = removedLeaves.copy()
			bestAvg = new_avg
			bestVerticesRemoved = verticesRemoved.copy()
			#print(str(new_avg) + "    " + "iteration " +  str(i))

		removedLeaves = []
		verticesRemoved = []
		currentAvg = avg
		temp = res.copy()
	print("rando" + str(bestAvg))
	return (bestLeaves, bestVerticesRemoved, bestAvg)



	

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

import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import random
from pathlib import Path
import numpy as np

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """

    vertices = list(G.nodes)
    edges = list(G.edges.data('weight'))

    best_tree = generateSolutionTwo(G)
    # res = KruskalMST(vertices, edges) #creates mst
    #
    # T = nx.Graph()
    # T.add_nodes_from(vertices)
    # T.add_weighted_edges_from(res)
    # avg_dist = average_pairwise_distance_fast(T) #current min pairwise avg of mst
    # n = list(T.nodes)
    # #base cases
    # if len(list(T.nodes)) == 1:
    # 	return T
    # if len(list(T.nodes)) == 2:
    # 	Te = nx.Graph()
    # 	Te.add_node(list(T.nodes)[0])
    # 	return Te


    # leaves = getLeaves(res)
    # if leaves != {}:
    # 	prunedEdges, prunedVertices, avgRand = pruneLeavesRando(leaves, res, avg_dist, vertices)
    # 	prunedEdgesD, prunedVerticesD, avgDesc = pruneLeavesDesc(leaves, res, avg_dist, vertices)
    # 	if avgDesc < avgRand:
    # 		T.remove_edges_from(prunedEdgesD)
    # 		T.remove_nodes_from(prunedVerticesD)
    # 	else:
    # 		T.remove_edges_from(prunedEdges)
    # 		T.remove_nodes_from(prunedVertices)
    #
    return best_tree

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
		input: edges
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

def pruneLeavesDesc(l, res, currentAvg, totalVs):
	'''
		input: l = all leaves of mst
			   res = complete mst
			   avg = mst's current avg
			   totalVs = total number of vertices in graph
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
		totalVs.remove(vertex)
		#print(temp)
		#create new graph to recalculate avg pairwise distance w/o elem
		Tr = nx.Graph()
		Tr.add_weighted_edges_from(temp)
		Tr.add_nodes_from(totalVs)

		new_avg = average_pairwise_distance_fast(Tr)
		#if better avg obtained w/o elem: remove it and update avg
		#else, add it back and move on to next leaf
		if new_avg <= currentAvg:
			removedLeaves.append(elem)
			verticesRemoved.append(vertex) #add vertices to this set
			currentAvg = new_avg
		else:
			temp.insert(i, elem)
			totalVs.append(vertex)
	#print("descAvg:" + str(currentAvg))
	return (removedLeaves, verticesRemoved, currentAvg)

def pruneLeavesRando(l, res, avg, totalV):
	'''
		input: l = all leaves of tree
			   res = tree's list of edges
			   avg = tree's current avg
			   totalV = tree's list of vertices
		output: (best edges, best vertices) of leaves to be removed
				from mst to minimize avg
	'''
	v = totalV.copy()
	removedLeaves = []
	temp = res.copy()
	#print("current mst: " + str(temp))
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
			#print("edge being removed: " + str(elem) + " at iteration: " + str(i))
			v.remove(vertex)
			#create new graph to recalculate avg pairwise distance w/o elem
			Tr = nx.Graph()
			Tr.add_nodes_from(v)
			Tr.add_weighted_edges_from(temp)
			#print("after edge removed, remaining nodes in T: " + str(Tr.nodes))
			#print("after edge removed, remaining edges in T: " + str(Tr.edges))
			if Tr.nodes == 0:
				new_avg = 0
			else:
				new_avg = average_pairwise_distance_fast(Tr)
			#if better avg obtained w/o elem: remove it and update avg
			#else, add it back and move on to next leaf
			if new_avg <= currentAvg:
				removedLeaves.append(elem)
				verticesRemoved.append(vertex) #add vertices to this set
				currentAvg = new_avg
			else:
				temp.insert(i, elem)
				v.append(vertex)
		if new_avg <= bestAvg:
			bestLeaves = removedLeaves.copy()
			bestAvg = new_avg
			bestVerticesRemoved = verticesRemoved.copy()
			#print(str(new_avg) + "    " + "iteration " +  str(i))

		removedLeaves = []
		verticesRemoved = []
		currentAvg = avg
		temp = res.copy()
		v = totalV.copy()
	#print("rando" + str(bestAvg))
	return (bestLeaves, bestVerticesRemoved, bestAvg)


def generateSolutionTwo(G):
    combination_of_edges = []
    vertices = list(G.nodes)
    num_of_vertices = len(vertices)
    edges = list(G.edges.data('weight'))
    sorted_edges = sortEdges(edges)
    num_of_edges = len(edges)
    k = num_of_vertices - 1
    best_tree = G
    best_avg_pairwise = float('inf')

    probabilities = [5]*(num_of_edges//5) + [4]*(num_of_edges//5) + [3]*(num_of_edges//5) + [2]*(num_of_edges//5) + [1]*(num_of_edges//5 + num_of_edges%5)
    # making this sum equal to 1
    probabilities /= np.sum(probabilities)
    totalTimeSteps = 100000

    while (totalTimeSteps > 0):
        # generates random indices on a distribution of sortedEdges array
        samples = np.random.choice(list(range(0,num_of_edges)), size = k, replace = False, p = probabilities)
        combination_of_edges = [sorted_edges[i] for i in samples]

        combination_of_verts = {combination_of_edges[0][0]}
        # checks if it's spanning
        for u,v,w in combination_of_edges:
            combination_of_verts.add(v)
            combination_of_verts.add(u)

        # if it is spanning, check if it's a tree
        if(len(combination_of_verts) == num_of_vertices):
            # solution_T = nx.Graph()
            # solution_T.add_nodes_from(list(G.nodes))
            # solution_T.add_weighted_edges_from(combination_of_edges)
        #
        #     # if it is a tree, prune
        #     if(is_Tree(solution_T)):
        #         # calculate initial pairwise distance
        #         current_pairwise_dist = average_pairwise_distance_fast(solution_T)
        #         leaves = getLeaves(list(solution_T.edges.data('weight')))
        #         if leaves != {}:
        #         	prunedEdges, prunedVertices, avgRand = pruneLeavesRando(leaves, solution_T, current_pairwise_dist, num_of_vertices)
        #         	prunedEdgesD, prunedVerticesD, avgDesc = pruneLeavesRando(leaves, solution_T, current_pairwise_dist, num_of_vertices)
        #         	if avgDesc < avgRand:
        #         		solution_T.remove_edges_from(prunedEdgesD)
        #         		solution_T.remove_nodes_from(prunedVerticesD)
        #         	else:
        #         		solution_T.remove_edges_from(prunedEdges)
        #         		solution_T.remove_nodes_from(prunedVertices)
        #
        #         print(totalTimeSteps)
        #         print(list(solution_T.edges.data('weight')))


            solution_T = nx.Graph()
            solution_T.add_nodes_from(vertices)
            solution_T.add_weighted_edges_from(combination_of_edges)

            # if it is a tree, we prune
            if(nx.is_tree(solution_T)):
                # calculate initial pairwise distance
                current_pairwise_dist = average_pairwise_distance_fast(solution_T)
                leaves = getLeaves(list(solution_T.edges.data('weight')))
                print(leaves)
                if leaves != {}:
                    prunedEdges, prunedVertices, avgRand = pruneLeavesRando(leaves, combination_of_edges, current_pairwise_dist, vertices)
                    prunedEdgesD, prunedVerticesD, avgDesc = pruneLeavesDesc(leaves, combination_of_edges, current_pairwise_dist, vertices)
                    better_avg = min(avgDesc,avgRand)
                    if better_avg < best_avg_pairwise:
                        if better_avg == avgDesc:
                            print("avgDist: " + str(avgDesc))
                            solution_T.remove_edges_from(prunedEdgesD)
                            solution_T.remove_nodes_from(prunedVerticesD)
                        else:
                            print("avgRand: " + str(avgRand))
                            solution_T.remove_edges_from(prunedEdges)
                            solution_T.remove_nodes_from(prunedVertices)
                        best_tree = solution_T
                        best_avg_pairwise = better_avg

                print("\n")
                print("totalTimeSteps: " + str(totalTimeSteps))

        totalTimeSteps = totalTimeSteps - 1
    return best_tree

def simulatedAnnealing(G):
    vertices = list(G.nodes)
    edges = list(G.edges.data('weight'))
    #generating random start vertex
    v = vertices[random.randint(0,len(vertices) - 1)]
    #starting our current tree with start vertex
    solution_T = nx.Graph()
    solution_T.add_node(v)

    curr_pairwise_dist = float("inf")
    totalTimeSteps = 10000

    # timesteps > 0 or until we've added all the vertices to the tree
    while (totalTimeSteps > 0  or len(list(solution_T.nodes)) < len(vertices)):
        # getting the minimum outgoing edge
        #outgoing_Edges = {4: {'weight': 25.643}, 1: {'weight': 54.048}}
        outgoing_edges = [(n, nbrdict) for n, nbrdict in G.adjacency() if n == v][0][1]
        degree_One = ()
        #outgoing_Edges = {(0,4):  25.643, (0,1): 54.048}
        outgoing_edges = {(v,x):y['weight'] for x,y in outgoing_edges.items()}
        #min_edge = (0,4)
        min_edge = min(outgoing_edges.keys(), key=(lambda k: outgoing_edges[k]))
        #(min_edge, outgoing_edges[min_edge]) = ((0,4), 25)
        min_edge = (min_edge, outgoing_edges[min_edge])
        # end vertex of min edge
        u = min_edge[0][1]

        temp = solution_T.copy()
        print(totalTimeSteps)
        print(list(temp.edges))
        temp.add_node(u)
        temp.add_edge(v, u, weight=min_edge[1])
        print(list(temp.edges))
        temp_pairwise_distance = average_pairwise_distance_fast(temp)

        if(temp_pairwise_distance < curr_pairwise_dist):
            try:
                nx.find_cycle(temp)
            except:
                solution_T.add_node(u)
                solution_T.add_edge(v, u, weight=min_edge[1])
                v = u
                curr_pairwise_dist = temp_pairwise_distance
        elif(math.exp((curr_pairwise_dist - temp_pairwise_distance) / totalTimeSteps) < random.uniform(0,1)):
            try:
                nx.find_cycle(temp)
            except:
                solution_T.add_node(u)
                solution_T.add_edge(v, u, weight=min_edge[1])
                v = u
                curr_pairwise_dist = temp_pairwise_distance

        totalTimeSteps *= 1-(0.003);
    return solution_T


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]
    G = read_input_file(path)
    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    write_output_file(T, 'outputs/test.out')

    # assert len(sys.argv) == 2
    # folder = sys.argv[1]
    # pathlist = Path(folder).glob('**/*.in')
    # for path in pathlist:
    # 	p = str(path).split("\\")[1].split(".")[0]
    # 	print(path)
    # 	G = read_input_file(path)
    # 	T = solve(G)
    # 	assert is_valid_network(G, T)
    # 	print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # 	write_output_file(T, 'outputs/' + str(p) + ".out")

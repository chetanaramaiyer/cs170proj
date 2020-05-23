import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import sys
import random
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as plt

from heapq import heappop, heappush
from operator import itemgetter
from itertools import count
from math import isnan
from datetime import datetime

import networkx as nx
from networkx.utils import UnionFind, not_implemented_for
from networkx import *

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """

    vertices = list(G.nodes)
    edges = list(G.edges.data('weight'))

    for u,v,w in edges:
        if(u == v):
            edges.remove((u,v,w))

    #[print(v) for v in vertices if G.degree[v] == 0]

    # BASE CASES
    # If there is one vertex
    if len(list(G.nodes)) == 1:
        one = nx.Graph()
        one.add_node(vertices[0])
        return one
    # If there two vertices
    if len(list(G.nodes)) == 2:
        two = nx.Graph()
        two.add_node(vertices[0])
        return two
    #If it is a complete graph
    if(len(edges) == ((len(vertices) * (len(vertices) - 1))/2)):
        complete = nx.Graph()
        complete.add_node(vertices[0])
        return complete
    #Checking if one vertex is connected to every other vertex
    for v in vertices:
        if(G.degree[v] == (len(vertices) - 1)):
            central = nx.Graph()
            central.add_node(v)
            return central

    #visualizeGraph(G, edges, vertices)
    best_tree_kruskals, best_avg_kruskals = generateSolutionOne(G, vertices, edges)
    print("Finished solution one")

    #if the graph = spanning tree, just run random pruning a couple of times
    if len(edges) == len(vertices) - 1:
        return best_tree_kruskals

    # Don't want to run solution two if MST found best possible pairwise distance
    if(best_avg_kruskals == 0):
        #print("MST 0")
        return T
    #best_tree_solution_two, best_avg_two = generateSolutionTwo(G)
    #print("Finished solution two")
    best_tree_three, best_avg_three = generateSolutionThree(G,vertices,edges)
    print("Finished solution three")

    final_best = min(best_avg_kruskals, best_avg_three)

    if(final_best == best_avg_kruskals):
        print("MST MST MST")
        return best_tree_kruskals
    # elif(final_best == best_avg_two):
    #     print("SOLUTION TWO")
    #     return best_tree_solution_two
    print("SOLUTION THREE")
    return best_tree_three

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
    result = []
    #This will store the resultant MST
    i = 0
    # An index variable, used for sorted edges
    e = 0
    # An index variable, used for result[]
    # Step 1: Sort all the edges in non-decreasing order of their weight. If we are not allowed to change the given graph, we can create a copy of graph
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
		# Step 2: Pick the smallest edge and increment the index for next iteration
        u,v,w = sortedEdges[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent ,v)

		# If including this edge does't cause cycle, include it in result and increment the index of result for next edge
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

def completePrune(T, leaves, edges, currentAvg, vertices):
    if leaves != {}:
        prunedEdges, prunedVertices, avgRand = pruneLeavesRando(leaves, edges, currentAvg, vertices)
        prunedEdgesD, prunedVerticesD, avgDesc = pruneLeavesDesc(leaves, edges, currentAvg, vertices)
        prunedEdgesAll, prunedVerticesAll, avgAll = pruneLeavesAll(leaves, edges, currentAvg, vertices)
        min_avg = min(avgRand, avgDesc, avgAll)
        if avgDesc == min_avg:
            T.remove_edges_from(prunedEdgesD)
            T.remove_nodes_from(prunedVerticesD)
            #print("desc")
        elif avgRand == min_avg:
            #print(prunedEdges)
            T.remove_edges_from(prunedEdges)
            T.remove_nodes_from(prunedVertices)
            #print("rand")
        else:
            T.remove_edges_from(prunedEdgesAll)
            T.remove_nodes_from(prunedVerticesAll)
            #print("all")
    return T, min_avg

def pruneLeavesAll(l, res, currentAvg, totalVs):
    prunedEdges = l.values()
    prunedVertices = l.keys()
    Tr = nx.Graph()
    Tr.add_weighted_edges_from([i for i in res if i not in prunedEdges])
    Tr.add_nodes_from([i for i in totalVs if i not in prunedVertices])
    avgAll = average_pairwise_distance_fast(Tr)
    return (prunedEdges, prunedVertices, avgAll)

def pruneLeavesDesc(l, res, currentAvg, totalVs):
    leaves = l
    removedLeaves = []
    temp = copy.deepcopy(res)
    verticesRemoved = []
    #start with the largest edge weight of leaves first
    def by_value(item):
        return item[1]
    for vertex, elem in sorted(leaves.items(), key=by_value, reverse=True):
        try:
            ind = temp.index(elem)
        except:
            elem = (elem[1], elem[0], elem[2]) #keep track of swapped lea
            ind = temp.index(elem)
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
            temp.insert(ind, elem)
            totalVs.append(vertex)
            #print("descAvg:" + str(currentAvg))
        return (removedLeaves, verticesRemoved, currentAvg)

def pruneLeavesRando(l, res, avg, totalV):
    '''input: l = all leaves of tree
    			   res = tree's list of edges
    			   avg = tree's current avg
    			   totalV = tree's list of vertices
    		output: (best edges, best vertices) of leaves to be removed
    				from mst to minimize avg
    '''
    v = copy.deepcopy(totalV)
    removedLeaves = []
    temp = copy.deepcopy(res)
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
    for i in range(0, len(l) * 5):
		#shuffle dictionary
        random.shuffle(keys)
        leaves = dict()
        for key in keys:
            leaves.update({key:l[key]})
        for vertex, elem in leaves.items():
			#ind = temp.index(elem)
            try:
                ind = temp.index(elem)
            except:
                elem = (elem[1], elem[0], elem[2]) #keep track of swapped lea
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
            if new_avg < currentAvg:
                removedLeaves.append(elem)
                verticesRemoved.append(vertex) #add vertices to this set
                currentAvg = new_avg
            else:
                temp.insert(i, elem)
                v.append(vertex)
        if new_avg < bestAvg:
            bestLeaves = removedLeaves.copy()
            bestAvg = new_avg
            bestVerticesRemoved = verticesRemoved.copy()
			#print(str(new_avg) + "    " + "iteration " +  str(i))
        removedLeaves = []
        verticesRemoved = []
        currentAvg = avg
        temp = copy.deepcopy(res)
        v = copy.deepcopy(totalV)
	#print("rando" + str(bestAvg))
    return (bestLeaves, bestVerticesRemoved, bestAvg)

def dummyTest(G):
    T = nx.Graph()
    T.add_node(list(G.nodes)[0])
    return T, float("inf")

def visualizeGraph(G, edges, vertices):
    # DRAWING GRAPH
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=50)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1)
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

    # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    plt.axis("off")
    plt.show()

def generateSolutionOne(G, vertices, edges):
    #creates mst
    res = KruskalMST(vertices, edges)
    T = nx.Graph()
    T.add_nodes_from(vertices)
    T.add_weighted_edges_from(res)
    #current min pairwise avg of mst
    kruskals_avg_dist = average_pairwise_distance_fast(T)
    kruskals_verts = list(T.nodes)
    kruskals_edges = list(T.edges)
    #base cases
    if len(kruskals_verts) == 1:
        return T
    if len(kruskals_verts) == 2:
        Te = nx.Graph()
        Te.add_node(kruskals_verts[0])
        return Te

    leaves = getLeaves(res)
    T, best_avg_mst = completePrune(T, leaves, res, kruskals_avg_dist, kruskals_verts)
    #print("best avg mst of kruskals: " + str(best_avg_mst))
    return T, best_avg_mst

def generateSolutionTwo(G):
    random.seed(datetime.now())
    combination_of_edges = []
    vertices = list(G.nodes)
    num_of_vertices = len(vertices)
    edges = list(G.edges.data('weight'))
    sorted_edges = sortEdges(edges)
    num_of_edges = len(edges)
    k = num_of_vertices - 1
    best_tree = G
    best_avg_pairwise = 5000000

    probabilities = [5]*(num_of_edges//5) + [4]*(num_of_edges//5) + [3]*(num_of_edges//5) + [2]*(num_of_edges//5) + [1]*(num_of_edges//5 + num_of_edges%5)
    # making this sum equal to 1
    probabilities /= np.sum(probabilities)
    #print(probabilities)
    totalTimeSteps = 100000
    num_of_lowest_appeareances = 0
    graphCount = 0

    while (totalTimeSteps > 0 and graphCount < 75):
        # generates random indices on a distribution of sortedEdges array
        samples = np.random.choice(list(range(0,num_of_edges)), size = k, replace = False, p = probabilities)
        combination_of_edges = [sorted_edges[i] for i in samples]

        combination_of_verts = {combination_of_edges[0][0]}
        # collects vertices in generated edges
        for u,v,w in combination_of_edges:
            combination_of_verts.add(v)
            combination_of_verts.add(u)

        # checks if it is spanning
        if(len(combination_of_verts) == num_of_vertices):
            vertices = list(G.nodes)
            solution_T = nx.Graph()
            solution_T.add_nodes_from(vertices)
            solution_T.add_weighted_edges_from(combination_of_edges)

            # checks if it is a tree
            if(nx.is_tree(solution_T)):
                # it is a valid graph, so increase graph count
                graphCount += 1
                # calculate initial pairwise distance
                current_pairwise_dist = average_pairwise_distance_fast(solution_T)
                leaves = getLeaves(list(solution_T.edges.data('weight')))
                #print("Total time steps: " + str(totalTimeSteps) + "\n")
                #print("Graph Count" + str(graphCount))
                if(leaves != {}):
                    prunedEdges, prunedVertices, avgRand = pruneLeavesRando(leaves, combination_of_edges, current_pairwise_dist, vertices)
                    #print("avgRand: " + str(avgRand))
                    prunedEdgesD, prunedVerticesD, avgDesc = pruneLeavesDesc(leaves, combination_of_edges, current_pairwise_dist, vertices)
                    #print("avgDesc: " + str(avgDesc))
                    prunedEdgesAll, prunedVerticesAll, avgAll = pruneLeavesAll(leaves, combination_of_edges, current_pairwise_dist, vertices)
                    #print("avgAll: " + str(avgAll))
                    better_avg = min(avgRand, avgDesc, avgAll, current_pairwise_dist)
                    # Tracking the number of times the same min is computed
                    if(round(better_avg, 2) == round(best_avg_pairwise,2)):
                        num_of_lowest_appeareances+=1
                    if better_avg < best_avg_pairwise:
                        if avgDesc == better_avg:
                            better_avg = avgDesc
                            solution_T.remove_edges_from(prunedEdgesD)
                            solution_T.remove_nodes_from(prunedVerticesD)
                        elif avgRand == better_avg:
                            better_avg = avgRand
                            solution_T.remove_edges_from(prunedEdges)
                            solution_T.remove_nodes_from(prunedVertices)
                        elif avgAll == better_avg:
                            better_avg = avgAll
                            solution_T.remove_edges_from(prunedEdgesAll)
                            solution_T.remove_nodes_from(prunedVerticesAll)
                        else:
                            better_avg = current_pairwise_dist
                        best_tree = solution_T
                        best_avg_pairwise = better_avg
                        num_of_lowest_appeareances = 0
                        #print(best_avg_pairwise)
        # If the minimum appears 20 times
        if((100000-totalTimeSteps) >= 1000 and num_of_lowest_appeareances >= 20):
            #print("lowest appeared 20 times")
            return best_tree, best_avg_pairwise
        totalTimeSteps = totalTimeSteps - 1
    return best_tree, best_avg_pairwise

def generateSolutionThree(G, vertices, edges):
    # calculating heuristics for every vertex
    heuristics = {}
    for v in vertices:
        # getting the weights of outgoing edges --> 0 if there's no edge
        out_going_weights = nx.adjacency_matrix(G).todense()[v]
        # summing the outgoing weights
        sum_of_outgoing = np.sum(out_going_weights)
        # getting the degree of vertex
        num_of_outgoing = len(list(G.neighbors(v)))
        # calculating the average weight of outgoing edges
        average_outgoing_weight = (sum_of_outgoing) / (num_of_outgoing)
        # heuristic depends on weight and outdegree
        # dividing 1 by this so that heuristic for neighbors that don't have edges is inf
        heuristics[v] = (num_of_outgoing)/(average_outgoing_weight)
    # sorting potential start nodes in ascending order by their scores
    starts = sorted(heuristics, key=heuristics.get, reverse=True)
    final_graph = None
    final_avg_pairwise = 100000
    graphCount = 0
    # trying new graphs at every start vertex
    for s in starts:
        visited_so_far = [False for i in vertices]

        solution_T = nx.Graph()
        solution_T.add_node(s)
        visited_so_far[s] = True

        # marking all adjacent neighbors of start vertex as visited
        for v in list(G.neighbors(s)):
            visited_so_far[v] = True

        best_curr_avg_pairwise = 10000
        # WITHIN ONE GRAPH
        while(len(solution_T.nodes) < len(vertices)):
            best_crossing_edge = None
            best_crossing_weight = 100000
            best_crossing_heuristic = 100000
            pairwise_averages = []

            # Looking at every vertex in our current tree
            for v in list(solution_T.nodes):
                #print("New vertex \n")
                #print("V: " + str(v))
                # Looking at potential crossing edges
                for n in list(G.neighbors(v)):
                    #print("N: " + str(n))
                    visited_so_far[n] = True
                    # As long as the edge is to another vertex already in the tree
                    if(n not in list(solution_T.nodes)):
                        curr_crossing_weight = G[v][n]['weight']
                        curr_crossing_edge = (v, n)
                        curr_crossing_edge = (curr_crossing_edge[0],curr_crossing_edge[1], curr_crossing_weight)

                        if (len(vertices) <= 50):
                            temp_solution = nx.Graph(solution_T)
                            temp_solution.add_node(n)
                            temp_solution.add_edge(curr_crossing_edge[0],curr_crossing_edge[1], weight=curr_crossing_weight)
                            temp_edges = list(temp_solution.edges.data('weight'))
                            temp_nodes = list(temp_solution.nodes)
                            #print(temp_nodes)
                            temp_avg_dist = average_pairwise_distance_fast(temp_solution)
                            pairwise_averages += [(n, curr_crossing_edge, temp_avg_dist)]
                        else:
                            curr_heuristic = curr_crossing_weight / heuristics[n]
                            #print("Current heuristic: " + str(curr_heuristic))
                            #print("Best heuristic: " + str(best_crossing_heuristic))
                            if (curr_heuristic < best_crossing_heuristic):
                                best_crossing_edge = curr_crossing_edge
                                best_crossing_heuristic = curr_heuristic
                                best_crossing_weight = curr_crossing_weight
                                #print(best_crossing_edge)

            if(len(vertices) <=50):
                # Getting the info of the min crossing edge
                #print(pairwise_averages)
                min_pairwise_weight = min([w for v,e,w in pairwise_averages])
                new_info = [(v,e) for v,e,w in pairwise_averages if w == min_pairwise_weight]
                solution_T.add_node(new_info[0][0])
                solution_T.add_edge(new_info[0][1][0], new_info[0][1][1], weight=new_info[0][1][2])
                best_curr_avg_pairwise = min_pairwise_weight

            else:
                #print("ADDING EDGE " + str(len(solution_T.edges)) + ": " + str(best_crossing_edge))
                solution_T.add_node(best_crossing_edge[1])
                solution_T.add_edge(best_crossing_edge[0],best_crossing_edge[1], weight = best_crossing_edge[2])
                #best_curr_avg_pairwise = average_pairwise_distance_fast(solution_T)

        leaves = getLeaves(list(solution_T.edges.data('weight')))
        if(len(vertices) > 50):
            best_curr_avg_pairwise = average_pairwise_distance_fast(solution_T)
        pruned_tree, pruned_best_avg = completePrune(solution_T, leaves, list(solution_T.edges.data('weight')), best_curr_avg_pairwise, list(solution_T.nodes))
        #print("Average of Graph " + str(graphCount) + ": " + str(best_curr_avg_pairwise))

        # After generating one graph
        if(best_curr_avg_pairwise < final_avg_pairwise):
            final_graph = solution_T
            final_avg_pairwise = best_curr_avg_pairwise
        elif(pruned_best_avg < final_avg_pairwise):
            final_graph = pruned_tree
            final_avg_pairwise = pruned_best_avg
        if(final_avg_pairwise == 0):
            return final_graph, final_avg_pairwise
        #print("\n")
        #print("NEW STARTING POINT")
        graphCount+=1


    #print(list(final_graph.nodes))
    #print(list(final_graph.edges.data("weight")))
    #print("FINAL OF THREE: " + str(final_avg_pairwise))
    return final_graph, final_avg_pairwise

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

if __name__ == '__main__':
    # assert len(sys.argv) == 2
    # path = sys.argv[1]
    # G = read_input_file(path)
    # T = solve(G)
    # assert is_valid_network(G, T)
    # print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    # write_output_file(T, 'outputs/test.out')
    assert len(sys.argv) == 2
    folder = sys.argv[1]
    pathlist = Path(folder).glob('**/*.in')
    for path in pathlist:
        p = str(path).split("large/")[1].split(".")[0]
        print(path)
        G = read_input_file(path)
        T = solve(G)
        assert is_valid_network(G, T)
        print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        write_output_file(T, 'outputs/' + str(p) + ".out")

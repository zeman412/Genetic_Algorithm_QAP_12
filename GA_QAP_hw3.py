# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:03:34 2017

@author: zeman
"""

import numpy as np
from math import *

dist_matrix = np.array(
    [[0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5],
     [1, 0, 1, 2, 2, 1, 2, 3, 3, 2, 3, 4],
     [2, 1, 0, 1, 3, 2, 1, 2, 4, 3, 2, 3],
     [3, 2, 1, 0, 4, 3, 2, 1, 5, 4, 3, 2],
     [1, 2, 3, 4, 0, 1, 2, 3, 1, 2, 3, 4],
     [2, 1, 2, 3, 1, 0, 1, 2, 2, 1, 2, 3],
     [3, 2, 1, 2, 2, 1, 0, 1, 3, 2, 1, 2],
     [4, 3, 2, 1, 3, 2, 1, 0, 4, 3, 2, 1],
     [2, 3, 4, 5, 1, 2, 3, 4, 0, 1, 2, 3],
     [3, 2, 3, 4, 2, 1, 2, 3, 1, 0, 1, 2],
     [4, 3, 2, 3, 3, 2, 1, 2, 2, 1, 0, 1],
     [5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0]])

flow_matrix = np.array(
    [[0, 5, 2, 4, 1, 0, 0, 6, 2, 1, 1, 1],
     [5, 0, 3, 0, 2, 2, 2, 0, 4, 5, 0, 0],
     [2, 3, 0, 0, 0, 0, 0, 5, 5, 2, 2, 2],
     [4, 0, 0, 0, 5, 2, 2, 10, 0, 0, 5, 5],
     [1, 2, 0, 5, 0, 10, 0, 0, 0, 5, 1, 1],
     [0, 2, 0, 2, 10, 0, 5, 1, 1, 5, 4, 0],
     [0, 2, 0, 2, 0, 5, 0, 10, 5, 2, 3, 3],
     [6, 0, 5, 10, 0, 1, 10, 0, 0, 0, 5, 0],
     [2, 4, 5, 0, 0, 1, 5, 0, 0, 0, 10, 10],
     [1, 5, 2, 0, 5, 5, 2, 0, 0, 0, 5, 0],
     [1, 0, 2, 5, 1, 4, 3, 5, 10, 5, 0, 2],
     [1, 0, 2, 5, 1, 0, 3, 0, 10, 0, 2, 0]])

size_population = 50
chromosome_len = 12            
g_max = 10  
Pm = 0.02
Pc = 0.5
np.random.seed(12345678)

def cros_over(parent_1, parent_2, cros_idx1, cros_idx2):
    if cros_idx1 > cros_idx2:
        cros_idx1,cros_idx2 = cros_idx2,cros_idx1

    n = chromosome_len

    index = list(range(cros_idx2+1,n)) + list(range(0, cros_idx1))
    child_1, child_2 = parent_1.copy(), parent_2.copy()
  
    child_1["chromosm"][index] = [p for p in parent_2["chromosm"] if p not in parent_1["chromosm"][cros_idx1:cros_idx2+1]]
    child_2["chromosm"][index] = [p for p in parent_1["chromosm"] if p not in parent_2["chromosm"][cros_idx1:cros_idx2+1]]

    child_1["cost"], child_2["cost"] = cost(child_1["chromosm"]), cost(child_2["chromosm"])

    return child_1, child_2

def mutatation_swap(individual, idx_1, idx_2):
    if idx_1 > idx_2:
        idx_1,idx_2 = idx_2,idx_1

    individual["cost"] = mutation_cost(individual["chromosm"], individual["cost"], idx_1, idx_2)
    individual["chromosm"][idx_1], individual["chromosm"][idx_2] = individual["chromosm"][idx_2], individual["chromosm"][idx_1]

def ga_run_main():

    length = chromosome_len
    size_pop = size_population

    solutions = np.zeros(g_max, dtype = np.int64)

    num_crosses = ceil(size_pop/2.0 * Pc)
    num_mutations = ceil(size_pop * length * Pm)
    #print (num_crosses)       #20
    #print (num_mutations)     #12

    len_chrom_str = str(length) + 'int'
    #print (len_chrom_str)
    datType = np.dtype([('chromosm', len_chrom_str), ('cost', np.int64)])
    
    # Generate initial population
    parents = np.zeros(size_pop, dtype = datType)   # Numpy empty container
    
    for indiv_soln in parents:
        indiv_soln["chromosm"] = np.random.permutation(length)
        indiv_soln["cost"] = cost(indiv_soln["chromosm"])

    parents.sort(order = "cost", kind = 'mergesort')
    print (parents)
    for g in range(g_max):
        
       # Tournament selection
        contestant_Idx = np.empty(size_pop, dtype = np.int32)  # contestants in tournament
        
        for index in range(size_pop):
            contestant_Idx[index] = np.random.randint(low = 0, high = np.random.randint(1, size_pop))
            
        contestant_pairs = zip(contestant_Idx[0:2*num_crosses:2], contestant_Idx[1:2*num_crosses:2]) 
        print (contestant_pairs)
        # Crossover the selected chromosomes
        children = np.zeros(size_pop, dtype = datType)  
        cross_points = np.random.randint(length, size = 2*num_crosses)
        
        for pairs, index1, index2 in zip(contestant_pairs, range(0,2*num_crosses,2), range(1,2*num_crosses,2)):
            children[index1], children[index2] = cros_over(parents[pairs[0]], parents[pairs[1]], cross_points[index1], cross_points[index2])
            #print (index1)
        children[2*num_crosses:] = parents[contestant_Idx[2*num_crosses:]].copy()

        # Mutate the children 
        mutant_children_Idx = np.random.randint(size_pop, size = num_mutations)
        mutant_allele = np.random.randint(length, size = 2*num_mutations)

        for index, gen_Index in zip(mutant_children_Idx, range(num_mutations)):
            mutatation_swap(children[index], mutant_allele[2*gen_Index], mutant_allele[2*gen_Index+1])

        # Replace the parents with children
        children.sort(order = "cost", kind = 'mergesort')
        
        #1. with elitism -- it gets to the optimal faster
        if children["cost"][0] > parents["cost"][0]:
           children[-1] = parents[0]
         #2. replace without elitism
        parents = children  # replace the parent with children

        parents.sort(order = "cost", kind = 'mergesort')
        #print("Best solution", g, ":", parents[0]["chromosm"])
        print("cost of best solution in the generation", g, ":", int(parents[0]["cost"]))
        solutions[g] = parents[0]["cost"]

    return parents[0]["chromosm"], parents[0]["cost"]

def mutation_cost(current_chromosom, current_cost, idx_1, idx_2):
    new_chromosom = np.copy(current_chromosom)
    new_chromosom[idx_1], new_chromosom[idx_2] = new_chromosom[idx_2], new_chromosom[idx_1]

    mutationCost = current_cost - sum(np.sum(flow_matrix[[idx_1,idx_2], :] * dist_matrix[current_chromosom[[idx_1,idx_2], None], current_chromosom], 1))
    mutationCost += sum(np.sum(flow_matrix[[idx_1,idx_2], :] * dist_matrix[new_chromosom[[idx_1,idx_2], None], new_chromosom], 1))

    idx = list(range(chromosome_len)) ; del(idx[idx_1]) ; del(idx[idx_2-1])

    mutationCost -= sum(sum(flow_matrix[idx][:,[idx_1,idx_2]] * dist_matrix[current_chromosom[idx, None], current_chromosom[[idx_1,idx_2]]]))
    mutationCost += sum(sum(flow_matrix[idx][:,[idx_1,idx_2]] * dist_matrix[new_chromosom[idx, None], new_chromosom[[idx_1,idx_2]]]))

    return mutationCost

def cost(chromosome):
    return sum(np.sum(flow_matrix * dist_matrix[chromosome[:, None], chromosome], 1))

if __name__== "__main__":  # calling the main function, where the program starts running 
  ga_run_main()
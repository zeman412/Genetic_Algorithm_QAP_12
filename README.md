# Implementation of genetic algorithm to solve the Quadratic Assignment Problem (QAP).
This code is an implementation of genetic algorithm to solve the Quadratic Assignment Problem (QAP) test problems 
of Nugent et al (12 department). The objective is to minimize flow costs between the placed departments. The 
flow cost is (flow * distance), where both flow and distance are symmetric between any given pair of departments. 

The implementation achieves the optimal solution provided as a benchmark for this problem. 
Using double flow to calculate the cost, the final solution is 578. And the optimal assignment 
vector is: [12,7,9,3,4,8,11,1,5,6,10,2].

In GA crossover is the main operator, hence the crosover probability should generally be set higher, close to 100%.
Several experiments should be done to evaluate the performance of the GA by changing the parameters for the GA.   


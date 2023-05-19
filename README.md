# <a name="_xg0ajysrsiq3"></a>Randomized Optimization

The first part of this project is to apply four search techniques - randomized hill climbing, simulated annealing, genetic algorithm, and MIMIC to three optimization problems to highlight different algorithm’s advantages. The second part is to use the first three algorithms to find good weights for a neural network of a problem in the project Supervised Learning.
# <a name="_84n1icm8xxx3"></a>Part 1
The performance of an algorithm on a problem always depends on the problem nature and the parameters applied. The project will go through the selected problems in sections. In each section there will be experiments identifying the best parameter set of each algorithm, followed by a comparison between all algorithms with the best found parameter for the problem.
## <a name="_oun1x2ojm4x8"></a>Problem 1 - FlipFlop
FlipFlop problem counts the number of flipped bits in a bit string and tries to maximize the flipped bits. It is a simple problem with random values on different bits, its fitness training should be well demonstrated by random based algorithms like Random Hill Climb and Simulated Annealing.

The first experiment is to examine the effect of maximum attempts on random hill climb. As shown in figure 1.1, max\_attempts should be set to at least 100 to allow the random search obtaining the best fitness with sufficient iteration. A low max\_attempts would rather result in a local maxima, since more random attempts are required to hit the correct state.

The second experiment is examining the population size and mutation probability on simulated annealing, shown in figure 1.2. When decays=0.99 the slow cool down rate makes it too long to obtain sub-optimal state. A decay-rate lower than 0.7 makes it able to obtain the best fitness faster.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.001.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.002.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.003.png)|
| :-: | :-: | :-: |
|Figure 1.1 |Figure 1.2 |Figure 1.3|

The third experiment is working on genetic algorithm. Figure 1.3 shows the mutation probability may slightly affect the fitness score and the converging rate. Figure 1.4 shows a population at 200 can obtain its optimal state, fewer or higher populations may end up converging to suboptimal states.

|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.004.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.005.png)|
| :-: | :-: |
|Figure 1.4|Figure 1.5|

Finally the experiment on MIMIC has shown that similarly a population at 200 is performing the best, as shown in figure 1.5. Provided with a right population size, a higher sample keep percentage would end up better result in figure 1.6.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.006.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.007.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.008.png)|
| :-: | :-: | :-: |
|Figure 1.6|Figure 1.7a|Figure 1.7b|

Figure 1.7a shows the performance of all algorithms. MIMIC behaves similarly to Genetic Algorithm, both require fewer iterations to converge. On the other side, Simulated Annealing and Random Hill Climb improves its fitness through the random search. RHC apparently stuck at local maxima but SA successfully obtains a better fitness with its changing temperature. Figure 1.7b shows the fitness score versus number of fitness evaluations. Despite lower iterations being taken in GA and MIMIC, they indeed do need more calling  the fitness function due to its large population, thus using more time.

Table 1 below compares the best fitness, number of iterations and runtime of the algorithms. While the number of iterations required is lower with Genetic Algorithm, it is taking much more time for each iteration and overall runtime, as it is working with a large population at each iteration. The situation is escalated in MIMIC as the algorithm is capturing the large pool of population and runs an estimation of distribution on each iteration.

FlipFlop problem has multiple local maxima. The experiments have shown that RHC fails to succeed. GA and MIMIC are doing better with their evolution on previous data but could only obtain an approximate global maxima. It is Simulated Annealing doing the best with its dynamic temperature and randomness so the optimal state can be achieved.


||**RHC**|**SA**|**GA**|**MIMIC**|
| :- | :- | :- | :- | :- |
|**Best Fitness**|42|49|46|47|
|**Min. Iteration**|119|846|37|16|
|**Runtime**|0\.034195|0\.272559|1\.661465|53\.386038|

Table 1 - Comparison of all algorithms over FlipFlop

I have also run experiments charting how algorithms behave with different problem sizes. They show a similar trend with RHC and SA growing closely, GA and MIMIC are similar. Note for the smallest size, GA and MIMIC can actually achieve the same fitness as SA at fewer iterations.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.009.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.010.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.011.png)|
| :-: | :-: | :-: |
|Figure 1.8|Figure 1.9|Figure 1.10|
##
## <a name="_wxn3s5ukupol"></a><a name="_uwfgj8hylf9q"></a>Problem 2 - Knapsack
Knapsack problem is briefly described as, given a set of items, each with a weight and a value, determine the number of each item to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible. The value of each node depends on other nodes, the interrelationship between nodes has made it a NP-hard problem. Compared with the first problem OneMax, Knapsack is a more complex problem that cannot be solved by simple random attempts, also more iterations are required intuitively.

Figure 2.1 has verified that by showing a higher number of maximum attempts, at least 50 is needed to reach the best fitness in RHC. Figure 2.2 shows SA can reach the optimal fitness faster with its reducing temperature. The decay rates however does not affect the performance, probably the local maxima are more separated.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.012.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.013.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.014.png)|
| :-: | :-: | :-: |
|Figure 2.1 |Figure 2.2 |Figure 2.3|


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.015.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.016.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.017.png)|
| :-: | :-: | :-: |
|Figure 2.4|Figure 2.5|Figure 2.6|

Figure 2.3 and 2.4 show the performance of Genetic Algorithm with different parameter sets. Initially a larger population size has gained better fitness but after iterations different population sizes are able to reach equal optimal fitness. Also, the same fitness result is achieved with different mutation probability. Apparently GA is doing well regardless of parameters, and is generally performing better than RHC. From my understanding, the evenly distributed population has evolved over successive generations, giving it a better chance to reach the optimal solution than RHC and SA that may have probably got stuck in local maxima.

Figure 2.5 and 2.6 shows the result of MIMIC algorithm over KnapSack. Similar to the previous experiment on OneMax, a larger population size has helped achieve optimal fitness. And a lower keep percentage under 0.2 is doing better. 

Figure 2.7a shows an overall comparison. The two evolutionary algorithms are significantly doing better, obtaining the best solution at the global maxima. Table 2 shows GA takes much longer time than RHC and SA, yet its significantly better result has made it the best algorithm in KnapSack. Intuitively the crossover process is able to achieve a better fitness over each iteration by looking over all items. Besides, MIMIC is also able to reach a satisfactory comparable result with its evolution from the previous iteration in the bounded search space. However its complex structure has consumed too much time, made it a worse algorithm than GA here. Besides, figure 2.7b demonstrated how the fitness changes versus evaluations. It is not surprising to see comparable results between GA and MIMIC across different stages of evaluations, it is the final runtime favoring GA as a better algorithm in this problem.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.018.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.019.png)|
| :-: | :-: |
|Figure 2.7a|Figure 2.7b|


||**RHC**|**SA**|**GA**|**MIMIC**|
| :- | :- | :- | :- | :- |
|**Best Fitness**|451|454|651|647|
|**Min. Iteration**|43|21|18|2|
|**Runtime**|0\.006117|0\.003405|0\.953222|2\.461244|

Table 2 - Comparison of all algorithms over KnapSack


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.020.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.021.png)|
| :-: | :-: |
|Figure 2.8|Figure 2.9|

Figure 2.8 and 2.9 show charting how algorithms behave with different problem sizes. As the previous problem, the charts show algorithms trending as pairs. GA and MIMIC consistently perform well. 
## <a name="_3hjsqeuf09rl"></a>Problem 3 - Queens
In this particular Queens problem, it consists of finding positions for 50 queens on an 50x50 square board so that the number of pairs of attacking queens is represented by the fitness. It is a famous NP problem with O(n^n) time complexity. As the problem can be represented well by structure, I would like to highlight the advantage of MIMIC. In the following experiments, there will also be assumption that the computation of each iteration is expensive, so the maximum number of iterations is limited to 40. 

From figure 3.1, RHC can at most reach a fitness of 82, the brute-force method does not work well here. Figure 3.2 shows SA has similar progress as RHC. Figure 3.3 and 3.4 shows GA is improving along with more iterations under different settings, its exchanging information is able to make the broad to a better state; it is strong in this type of problem. 


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.022.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.023.png)|
| :-: | :-: |
|Figure 3.1 |Figure 3.2 |


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.024.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.025.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.026.png)|
| :-: | :-: | :-: |
|Figure 3.3|Figure 3.4|Figure 3.5|


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.027.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.028.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.029.png)|
| :-: | :-: | :-: |
|Figure 3.6|Figure 3.7|Figure 3.8|

The problem Queens depends on the states of nodes and structures. I expected it could be well handled by MIMIC. The experiment has verified this with the population size at 250 and keep percentage at 0.25, as shown in figure 3.5 and 3.6. Figure 3.7a shows an overall comparison between all algorithms, with MIMIC performing the best with a fitness score of 96. Figure 3.8 shows even all algorithms are bound to 40 iterations, GA and MIMIC has run more than 100 times fitness evaluations to achieve better fitness. When the same evaluation number is limited, similar performances are found across the algorithms. This experiment highlights the strength of MIMIC when iteration computation but not evaluation computation is expensive.


||**RHC**|**SA**|**GA**|**MIMIC**|
| :- | :- | :- | :- | :- |
|**Best Fitness**|82|77|93|97|
|**Min. Iteration**|27|13|34|9|
|**Runtime**|0\.096548|0\.062844|31\.887019|34\.679336|

Table 3 - Comparison of all algorithms over Queens

Table 3 shows the performances in detail. The evolutionary algorithms GA and MIMIC are doing better in general. My understanding is the information exchange from a set of populations is gradually working to a better state for the overall solution. MIMIC has shown by calculating probability distribution over each parameter, it is possible to obtain the optimal state with fewer than one third iterations than that of GA. The overall runtime on my local computer is only 10% more than GA’s, which is a reasonable trade-off for better fitness.

Experiments below charting how algorithms behave with different problem sizes. It is noted that when the problem size is low, GA and MIMIC are performing similarly well. Only when the problem is set to at least the size of 15 (in figure 2.7), we can see a difference when iteration is limited to 10. And the previous experiment with a size of 50 highlighted how better MIMIC can do.

|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.030.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.031.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.032.png)|
| :-: | :-: | :-: |
|Figure 3.8|Figure 3.9|Figure 3.10|
## <a name="_6a9w43s7lws1"></a>Conclusion of Part 1
From the experiments above, it is found that for simple problems Random Hill Climb can obtain a good fitness at a fair speed with its randomness nature. Sometimes it could be hard to converge, or get out of local maxima if the base of reaching global maxima is narrow. Simulated Annealing shows similar behavior as RHC, yet is sometimes better in avoiding local maxima when global maxima is far apart, and could require fewer iterations to converge, provided the correct parameter is given.

The evolutionary algorithms Genetic Algorithm and MIMIC take advantage of initial population  and evolution based on previous iteration, could possibly have a breakthrough on the fitness score on complex NP-hard problems especially those with structure. As a trade off, its runtime is usually much higher because of the computation on the pool of populations. The use of mutation in GA has successfully avoided local maxima, but its convergence rate could be rather slow if relying on a low percentage of random mutation. MIMIC on the other side requires fewer iterations, with its estimation on probability density over the population, and it especially works well on structure problems with dependent nodes. Again its runtime is the trade off, and parameter tuning is always required.
# <a name="_uy7w09vu1z9u"></a>Part 2
In part 2, I am going to analyze the Online Shopper Intention data in project 1 with three algorithms to find good weights for a neural network. First I will check how the parameters would affect performances of Simulated Annealing and Genetic Algorithms. For SA, Figure 4.1a shows that different parameter settings of decay rate only slightly affect the results. Figure 4.1b shows training and testing has similar curves on different parameters, with higher testing score representing higher variance at this setting. Only when the decay rate is above 0.9 we can see a small drop in accuracy.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.033.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.034.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.035.png)|
| :-: | :-: | :-: |
|Figure 4.1a|Figure 4.1b|Figure 4.2a|


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.036.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.037.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.038.png)|
| :-: | :-: | :-: |
|Figure 4.2b|Figure 4.3a|Figure 4.3b|

For GA, Figure 4.2a, 4.2b, 4.3a, and 4.3b below also show that the algorithm has the same performance under different settings. A consistent higher testing accuracy indicates the model has a higher variance with this train size.

Next I will compare the accuracy and runtime of different algorithms. Also included is the chart of learning algorithm Multi-layer Perceptron (MLP) that trains using Backpropagation, tuned with the best parameter setting in project 1, to contrast the performances. The hidden layer is set to **[20, 20]** and the solver is set as **adam**. In the following experiments I have set a high iteration number to observe their performance in a long shot. Figure 4.4 and 4.5 shows Genetic Algorithm has the best accuracy among the three algorithms, yet its training time is more than ten times of others. Apparently it takes much longer time on information exchange and mutation among its population. Simulated Annealing and Random Hill Climb has shorter training time, but accuracy is also 0.1 score lower than that of GA. 

Possible reasons are that SA and RHC are trapped at local maxima, while GA has avoided it with iterations over populations; or, SA and RHC have resulted in overfitting. Figure 4.6 has shown the fitness score and how convergence occurs, a lower fitness (loss function) score represents a lower squared root error of the training set. The discontinued line of GA means there is no more improvement in terms of fitness, after its initial drastic improvement converging the mean of the population. When the number of iterations is growing, SA and RHC shows a slowly decreasing error rate. It is also noticed that the number of iterations is lower in GA, but the computation over population resulted in a longer processing time.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.039.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.040.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.041.png)|
| :-: | :-: | :-: |
|Figure 4.4|Figure 4.5|Figure 4.6|

From figure 4.7, 4.8, 4.9, 4.10, the learning curves of the algorithms are also shown here. They show consistent results as figure 4.4, MLPClassifier is doing significantly better, followed by GA, RHC, and lastly SA. Nevertheless a larger train size does not necessarily result in a better test score, probably due to the noise. Also figure 4.7, 4.8, 4.10 all show the curves initially start with the training score higher than testing score, showing models with higher bias. However figure 4.9 shows that the model of SA has a higher initial testing score, that could be due to coincidence and unbalanced dataset. From all these graphs we can see the score of training and testing tend to converge with a higher number of training samples, to a point at the balance between bias and variance. The parameter of mlrose library, **early\_stopping**, terminate the algorithms early if the loss is not improving, thus the models would not show higher variance.


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.042.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.043.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.044.png)|
| :-: | :-: | :-: |
|Figure 4.7|Figure 4.8 |Figure 4.9|


|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.045.png)|![](graph/75f6395f-ae3f-4d7e-8aad-dd059ce51a7a.046.png)|
| :-: | :-: |
|Figure 4.10|Figure 4.11|

Finally, I am comparing the overall metrics score in figure 4.11. The metrics score of precision, recall, and F1 are averaged with weights, because the dataset is unbalanced. 

Experiments have shown the Multi-layer Perceptron classifier using backpropagation is doing the best in terms of both runtime and accuracy, as it computes gradient descent with respect to weights of multiple layers in the network structure representing the dataset in the Online Shopper Intention problem. It is a problem that its derivatives can be mathematically calculated to find weights easier than using random algorithms. Besides, among the three randomized optimization algorithms, GA is doing better in general with significant higher accuracy and recall score. Yet its precision is slightly worse than that of SA and RHC. And GA’s f1 score balancing precision and recall is again scoring better among others.

For the use case of the dataset Online Shopper Intention, it is often used as marketing analysis and business model prediction. Accuracy, recall and F1 could be more important factors that can compensate for the slightly lower precision. Also the dataset is static and not time critical so a long training time should be acceptable, as a trade off of better score. In conclusion Genetic Algorithm is the best to handle the dataset in Neural Network, among the three algorithms in Part 1. However, MLPClassific is still doing much better in terms of overall accuracy and runtime in Neural Network.

## <a name="_x932szno6yil"></a>Running under Anaconda
1. The environment file for anaconda is environment.yml
1. By default, the program can be executed by the command ./python RandomizedOptimization.py
It will then 
	- run experiements over the problems FlipFlop, Knapsack, Queens, and generate all resulting graphs into ./graph
	- run experiement using three algorithms to find good weights for a neural network, for the problem in the project Supervised Learning. All graphs will be generated in ./graph
1. To enable or disable experiements of different problems, `main()` in RandomizedOptimization.py should be modified. A complete set of codes have been already written by default. So running directly would execute all steps in sequence. Note the process may take long, commenting out unwanted experiements would allow geenerating targeting graphs. For example, commenting the following code would run only FlipFlop experiement.

		experiment = FlipFlop()
		experiment.run()

		# experiment = Knapsack()
		# experiment.run()

		# experiment = Queens()
		# experiment.run()

		# data = loadData_1(encode_category = True)
		# nn = NeuralNetworks()
		# nn.analyze(data)
1. By default, all algorithms will perform a full analysis. In case of a rerun on a particular chart or statistics, it can be done by commenting out functions from `run()` or `analyze()` in [problem].py that are not required. For example, in `run()` of FlipFlop.py, commenting out first 4 lines will get the program generate only the overall performance comparison of different algorithms.

		# self.evaluateRandomHillClimb(problem, color_list)
		# self.evaluateSimulatedAnnealing(problem, color_list)
		# self.evaluateGeneticAlgorithm(problem, color_list)
		# self.evaluateMimic(problem, color_list)        
		self.comparePerfermance(problem, color_list)
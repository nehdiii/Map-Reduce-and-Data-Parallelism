# Map-Reduce-and-Data-Parallelism

Some Machine Learning problems are just too big to run on one machine, sometimes maybe you just have such a large amount of data, (for instance you have 100 million training examples) that you would not want to run all that through a single computer, irrespective of what algorithm you are using. To combat this problem, a different approach to large scale Machine Learning known as the “Map-Reduce” approach, With the idea of Map-Reduce we would be able to scale learning algorithms to large machine learning problems, much larger than it is possible with Batch Gradient Descent or Stochastic Gradient Descent.

# How it works:
Let us suppose that we have 10 million (10M) examples in our training set and we want to fit a linear regression model or a logistic regression model.

The Batch Gradient Descent learning rule has these 10M and the sum from i equals 1 through 10M, i.e all my training examples, is a computationally expensive step.
Let us say I want to denote my training examples by means of the (X,Y) 

In Map-Reduce we split the training set into convenient number of subsets. Assume that we have 10 computers or CPUs in the lab to run in parallel on my training set, so we shall split the data into 10 subsets.

Each of the subset has 1M examples for 10 different machines. Each of these 10 machine will now run the Batch Gradient Descent learning rule for their respective 1M examples.

For the first machine with the first 1M examples we are going to compute a variable which is equal to the gradient for the first 1M examples.

Similarly we are going to take the second subset of my data and send it to the second machine, which will use training examples (1M+1) through 2M and we will compute a similar variable.

The rest of the 8 machines will use the remaining 8 subsets of my training set. Hence, now each machine has to sum over 1M examples instead of 10M and so has to do only 1/10th of the work, thus would presumably be almost 10 times faster.

After all the machines have computed the respective gradients, they are sent to a centralized master server.

# Multi cores approach:
Sometimes Map-Reduce can be applicable to a single computer with multiple CPUs or CPUs with multiple computing cores.
Say we have a computer with 4 computing cores. 
So we may split the training set and send the subsets to different cores within a single computer and divide the work load.
Each of the cores carry out the sum over one quarter of our training set and the individual results are combined to get the summation over the entire training set.

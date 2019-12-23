# AI-learns-to-land-Moon-Lander-
In this project, I have implemented Genetic Algorithm to let the agent learn to land a Moon Lander at a specific co-ordinate relayed to it.

I have used a population of 70 and a deep learning approach. Here, we have two hidden layers with 40 neurons each on an average. 
You can use your own as well.

Crossover: Single Crossover/ Uniform Crossover seems to work quite well.

Elitism fails for this project. Hence, I moved ahead wtih a random set of parents between 10 to 40 to next generation to keep a healthy 
variation.

The designed Model has crossed 200 median scores!! Celebrations
Update: I accidentally chose the SoftMax activation function every layer. I corrected that and now only the hidden layers have ReLU and the output layer has Softmax activation function.

Here's the GIF of multiple landings of our Lander.

![alt text](https://1.bp.blogspot.com/-Lokb2RhmSmc/XfMTqSjntRI/AAAAAAAAE3E/cSfWXM8C0W8mKNd7-cHYkHHxHIbjEvlngCLcBGAsYHQ/s1600/WORK.gif)


In my previous post, I solved the CartPole environment using a genetic algorithm. However, in the Cartpole environment, we have low complexity, space, and states. Also, Catpole begins with a state of maximum reward i.e the rendering always starts from the perpendicular position which is what we want out the environment to do. So basically, our algorithm already begins with the best reward and hence, requires fewer iterations to find. In Lunar Lander, the best reward state has to be discovered by our algorithm. Most of the time the Lander starts from same position but rarely does it return to the landing pad which eventually returns the best rewards. With this project, I came to realize that for the genetic methodology, if you have a large space, you would require more exploration, more population and elitism won't work.

Why elitism won't work?

Bill Gates' son is an elite person. Bill Gates isn't. How?
Bill Gates or Steve Jobs have seen worst as well the best of their times. They have seen a failure of the product as well as the success of the company. They have in relative more experience. On the other hand, Bill Gates' son has seen money from the very beginning. He/She might not know the real value of hardship as none of them would have to toil like Jobs or Gates. So overall, we don't just need 'What to do' which is the policy of elites. We also need 'What not to do' which is the policy of non-elites. The experience of non-elites helps us to filter the elites' ones as well.

PS. Thomas Alva Edison was a damn nonelite person for sure as he discovered 10000 ways that don't work.

The result of our algorithm?

New Output with correct activation function (ReLU in the hidden layers and Softmax for the output layers ):
The Green colored plot is the median of our population. The Blue colored plot is the maximum of each generation.

The tweaks in GA relies on the basics.

Changes: Earlier I had mistakenly set Softmax for every layer. This time the input and the two hidden layers have the ReLU activation function and the output layer has the Softmax activation function. With this new model, I reached the median score of 200 with a mere 860 generations as compared to 1054 generations with an earlier version. Below is a comparison of dissolved graphs.



The Black plot is the plot of the old 'best' scores of each generation. The Red plot is of the new. It can be seen that ReLU had more trouble in maintaining consistency initially with the best scores but grasped with the higher scores fastly. The dark blue plot is of the new model and the cyan plot if of old model median scores. The new model converges fastly than the old version. Until 100 generations as in the plot, the scores were pretty much the same but after that, the slope bumps up.



Have a look at the best score a.k.a History in the above screenshot. We have a consistent above 300 scores. The average score is also above 200 i.e. 287.79

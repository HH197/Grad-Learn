
# ZINB-Grad: A Gradient Based Linear Model Outperforming Deep Models
scRNA-seq experiments are powerful, but they suffer from technical noise, dropouts, batch effects, and biases (See Link for more details). 

Many statistical and machine learning methods have been designed to overcome these challenges. In recent years, there has been a shift from traditional statistical models to deep learning models. But are deep models better for scRNA-seq data analysis? 

Published literature claimed that deep-learning-based models, such as scVI, outperform conventional models, such as ZINB-WaVE. Here, we used a novel optimization procedure combined with modern machine learning software packages to overcome the scalability and efficiency challenges inherited in traditional tools. We showed that our implementation is more efficient than both conventional models and deep learning models. 

We assessed our proposed model, ZINB-Grad, and compared it with [scVI](https://www.nature.com/articles/s41592-018-0229-2) and [ZINB-WaVE](https://www.nature.com/articles/s41467-017-02554-5), both developed at the UC Berkeley, using a set of benchmarks, including run-time, goodness-of-fit, imputation error, clustering accuracy, and batch correction. 

Our development shows that a conventional model optimized with the proper techniques and implemented using the right tools can outperform state-of-the-art deep models. It can generalize better for unseen data; it is more interpretable, and not surprisingly, it uses extremely fewer resources compared to deep models.


## ZINB-WaVE 

ZINB-WaVE is a generalized linear mixed model (GLMM) which captures the technical variability through two known random variables and maps the data onto a biologically meaningful low-dimensional representation through a linear transformation. ZINB-WaVE extracts low-dimensional representation from the data by considering dropouts, over-dispersion, batch effects, and the count nature of the data. 

<img  src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-017-02554-5/MediaObjects/41467_2017_2554_Fig1_HTML.png?as=webp"> 


In the above schematic view, $X$ and $V$ are known sample-level and gene-level covariates, respectively. $X$ can model wanted or unwanted variations. For instance, $X$ can be used to correct batch effects, and $V$ can model gene length or GC-content. This is how ZINB-WaVE performs normalization and batch effect correction all in one step.

$W$ and $\alpha$ are the unknown matrices that are used for dimension reduction. Columns of $W$ are essentially the latent space of ZINB-WaVE, and $\alpha$ is its corresponding matrix of regression parameters. The $O$ parameters are the known offsets for $\pi$ and $\mu$.  For more details, please refer to [ZINB-WaVE](https://www.nature.com/articles/s41467-017-02554-5). 

### ZINB-WaVE Bottleneck

The bottleneck of the ZINB-WaVE is in its estimation process, which includes several iterative procedures, such as ridge regression and logistic regression. It also contains SVD decompositions of big matrices (for large sample sizes) and the BFGS quasi-Newton method. Not surprisingly, the optimization procedure requires heavy computations and high-memory storage. Therefore, the ZINB-WaVE is doomed to be scalable to only a few thousand cells, and nowadays, applications require millions of samples to be efficiently analyzed. 

However, this limitation does not imply that the traditional linear models are suboptimal compared to deep learning-based models. Indeed, we will show that, by switching the optimization procedure from the conventional analytic way to a modern method, ZINB-WaVE is scalable to millions of cells and even more efficient and productive than its deep learning descendent, such as scVI.


## ZINB-Grad

Supported by most modern machine learning packages (e.g., Pytorch and Tensorflow), gradient descent and its variants are used to optimize deep neural networks and are among the most popular algorithms for optimization. By updating the parameters in the opposite direction of the objective function’s gradient, gradient descent finds local optimum values of the model’s parameters to minimize the objective function regardless of how complicated is the model. 

However, a neglected fact is that gradient descent may also be applied to “simpler” linear models when appropriate. Towards this line, we developed a gradient descent-based stochastic optimization process for the ZINB-WaVE to overcome the scalability and efficiency challenges inherited in its optimization procedure. We combined the new optimization method with modern statistical and machine learning libraries, which resulted in the ZINB-Grad, a gradient-based ZINB GLMM with GPU acceleration, high-performance scalability, and memory-efficient estimation.

## Results 






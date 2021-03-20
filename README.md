# XGBOOST - Amal S Kumar M190004MS 
There are 2 main types of decision tree ensemble:

Bagging <br />
Boosting <br />

In Bagging we combine predictors from multiple-decision trees through a majority voting mechanism.

In Boosting we build models sequentially by minimizing the error from previous models while increasing (or boosting) influence of high performance models.

If compare new employee hiring process by few recruitment specialists in a company (sequential interviews) with Boosting, here each next recruitment specialist is based his decision on the assesment of the candidate by the previous manager.This speed up hiring process, as unsuitable candidates are immediately eliminated.
XGBoost is an algorithm that has recently been dominating applied machine learning and it is an  implementation of gradient boosted decision trees designed for speed and performance. This algorithm goes by lots of different names such as gradient boosting, multiple additive regression trees, stochastic gradient boosting or gradient boosting machines.

Boosting is an ensemble technique where new models are added to correct the errors made by existing models. Models are added sequentially until no further improvements can be made. A popular example is the AdaBoost algorithm that weights data points that are hard to predict.

Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models. This approach supports both regression and classification predictive modeling problems.
Let see some of the advantages of XGBoost algorithm:

1. Regularization: XGBoost has in-built L1 (Lasso Regression) and L2 (Ridge Regression) regularization which prevents the model from overfitting. That is why, XGBoost is also called regularized form of GBM (Gradient Boosting Machine).

2. Parallel Processing: XGBoost utilizes the power of parallel processing and that is why it is much faster than GBM. It uses multiple CPU cores to execute the model.

3. Handling Missing Values: XGBoost has an in-built capability to handle missing values. When XGBoost encounters a missing value at a node, it tries both the left and right hand split and learns the way leading to higher loss for each node. It then does the same when working on the testing data.

4. Cross Validation: XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run. This is unlike GBM where we have to run a grid-search and only a limited values can be tested.

5. Effective Tree Pruning: A GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm. XGBoost on the other hand make splits upto the max_depth specified and then start pruning the tree backwards and remove splits beyond which there is no positive gain.

The biggest limitation is probably the black box nature. If you need effect sizes, XGBoost won’t give them to you (though some adaboost-type algorithms can give that to you). You’d have to derive and program that part yourself. Given the models that exist (like penalized GLMs), XGBoost wouldn’t be your go-to algorithm for those use cases.

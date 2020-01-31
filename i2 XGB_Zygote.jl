
### Using custom loss functions in XGBoost with Zygote
# Machine learning algorithms optimize a loss function. There are a lot of cases where the commonly available loss functions might be inadequate.
# Consider trying to detect cancer with a machine learning model. Early detection can increase the chances of survival.
# In this scenario, it is desirable when the model is penalized heavily for a false negative. This is an example of asymmetric cost. These problems require custom loss functions that can take the asymmetry into account.
#
# In this notebook, we will see how easy it is to formulate and use custom loss functions with Zygote and how it seamlessly blends with XGBoost.
#

using XGBoost
using Zygote

### Load data

# In this exercise we will load the mushroom dataset from https://archive.ics.uci.edu/ml/datasets/mushroom
# This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom. The features correspond to the shape, colour, type etc.
# We will use xgboost with a custom loss to predict if they are toxic or not


train_data_file = joinpath(dirname(dirname(pathof(XGBoost))), "data", "agaricus.txt.train")

train_data = DMatrix(train_data_file)


### Define xgboost parameters

param = ["max_depth"=>2, "eta"=>1];#max depth defines depth of each tree in the ensemble, eta is shrinkage parameter
watchlist  = [(train_data,"train")];#monitor performance
num_round = 2; #number of boosters

### Define loss function
# The custom loss function here is a weighted logistic loss, that penalizes false positive and false negative differently. The FN weight is 5 and FP weight is 2


σ(x) = 1/(1+exp(-x));
weighted_logistic_loss(x,y) = -5 .*y*log(σ(x)) - 1 .*(1-y)*log(1-σ(x));

### Gradient and Hessian
# For any loss function, xgboost should be provided with the gradient and hessian of the loss function with respect to prediction.
#
# The objective function that should be passed to xgboost should look like

    # function custom_objective(prediction,label)
    #     #calculate gradient and hessian
    #
    #     return gradient, hessian


#### You can either manually calculate the gradient and hessian

### Or better, use Zygote to automatically calculate the gradients



grad_logistic(x,y) = gradient(weighted_logistic_loss,x,y)[1];


hess_logistic(x,y) = gradient(grad_logistic,x,y)[1];

### Wrap them all together

#### Tie custom objective function with its gradient and hessian

function custom_objective(x::Vector{Float32}, train_data::DMatrix)
                  y = get_info(train_data, "label")
                  grad = grad_logistic.(x,y)
                  hess = hess_logistic.(x,y)
                  return grad,hess
                  end



#### Following function will be used to monitor error rate on training data


function evalerror(preds::Vector{Float32}, train_data::DMatrix)
                  labels = get_info(train_data, "label")
                  return ("error", sum((preds .> 0.0) .!= labels) / float(size(preds)[1]))
              end


### Train model


bst = xgboost(train_data, num_round, param=param,
        watchlist=watchlist,
        obj=custom_objective,
        feval=evalerror)

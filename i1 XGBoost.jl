cd(@__DIR__)
import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
pkg"precompile"

using XGBoost
using DataFrames, CSV
using Statistics
using StatsBase: countmap

df = DataFrame(CSV.File("data/titanic.csv"))

describe(df)

countmap(df.Embarked)

df.Embarked = coalesce.(df.Embarked, "S")

df.Embarked = map(x-> Dict("S"=>3, "C"=>2, "Q"=>1)[x], df.Embarked);

average_age = mean(skipmissing(df[!, :Age]))

df.Age = coalesce.(df.Age, average_age)

countmap(df.Sex)

gender_dict = Dict("male"=>1, "female"=>0)

df.Sex = map(x->gender_dict[x], df.Sex);

cols = [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked,]

x_train = convert(Matrix{Float32}, df[1:800, cols])

x_test = convert(Matrix{Float32}, df[801:end, cols]);

y_train = df.Survived[1:800];

y_test = df.Survived[801:end];

bst = xgboost(x_train,15,label=y_train, eta=0.3, max_depth =2,objective = "binary:logistic",eval_metric="auc")

y_pred = XGBoost.predict(bst, x_test)

accuracy = round(sum(Int.(y_pred.>0.5) .== y_test)/length(y_test)*100;digits=2);

println("Accuracy on test set = ", accuracy,"%")

XGBoost.save(bst, "mymodel.model")

bst = Booster(model_file="mymodel.model");

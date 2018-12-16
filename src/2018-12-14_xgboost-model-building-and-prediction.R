

#*****************************************************
# USING XGBOOST TO BUILD A MODEL AND PREDICT STUFF 
# 2018-12-15
# Nayef 
#*****************************************************


library(xgboost)
library(tidyverse)
library(caret)
library(pROC)

help(package = "xgboost")

# 1)loading data: ---------------

data("agaricus.train")
data("agaricus.test")

train <- agaricus.train
test <- agaricus.test

# data is in list form: 
str(train, max.level = 1)
# first element is a dgCMatrix with predictors as numeric
# second element is the vector of labels (response variable)

# extract first element of list: 
train$data %>% str
# class 'dgCMatrix' [package "Matrix"] with 6 slots
# dgCMatrix is the “standard” class for sparse numeric
# matrices in the Matrix package.


dim(train$data)  # 6513 rows, 126 cols 
dim(test$data)

dimnames(train$data)  # note that most of these are factors 

# Apparently, this is a "very small" dataset


# 2) Basic training with sparse matrix input: ---------------

# In a sparse matrix, cells containing 0 are not stored in
# memory. Therefore, in a dataset mainly made of 0, memory
# size is reduced. It is very usual to have such dataset.

bstSparse <- xgboost(data = train$data, 
                     label = train$label, 
                     max_depth = 2, 
                     eta = 1,  # control the learning rate: scale the contribution of each tree by a factor of 0 < eta < 1 when it is added to the current approximation. Used to prevent overfitting by making the boosting process more conservative.  
                     nthread = 2, 
                     nrounds = 2, 
                     objective = "binary:logistic")



# 2) Basic training with dense matrix input: ---------------
as.matrix(train$data) %>% str
dimnames(train$data)

bstDense <- xgboost(data = as.matrix(train$data), 
                    label = train$label, 
                    max_depth = 2, 
                    eta = 1, 
                    nthread = 2, 
                    nrounds = 2, 
                    objective = "binary:logistic")


# 3) Basic training with xgb.DMatrix input: ---------------

dtrain <- xgb.DMatrix(data = train$data, 
                      label = train$label)
# str(dtrain)

bstDMatrix <- xgboost(data = dtrain, 
                      max_depth = 2, 
                      eta = 1, 
                      nthread = 2, 
                      nrounds = 2, 
                      objective = "binary:logistic", 
                      verbose = 2)  # to show info about tree



# 4) Prediction on test data: -------------

pred <- predict(bstDMatrix, 
                test$data)

head(pred)
# note that we haven't done binary classification yet. These
# numbers give Prob(TRUE)


# 5) Transform predicted probs to binary decision: -----

# just set any prob > 0.5 to true

# todo: isn't there any way to optimize the selection of the
# threshold?

prediction <- as.numeric(pred > 0.5)

table(prediction)


# 6) Measuring model performance: -----------

# To measure the model performance, we will compute a simple
# metric, the average error.

err <- mean(prediction != test$label)  # 0.0217

# This metric is 0.02 and is pretty low: our yummly mushroom
# model works well!

# > confusion matrix using caret package: ------ 
confusionMatrix(factor(prediction), 
                factor(test$label))

# looks really good! 


# > ROC curve using pROC package: -------
?roc

roc_test <- roc(test$label, 
                prediction, 
                algorithm = 2)

# todo: algorithm 1 vs 2? 

plot(roc_test)
auc(roc_test)  # Area under the curve: 0.9785


# 7) Other notes:  -------- 

# > always try both tree boosting and linear boostin, to see
# which is best for your problem To implement linaer
# boosting, use arg: booster = "gblinear"



# 8) Feature importance: -------------
importance.matrix <- xgb.importance(model = bstDMatrix)

xgb.plot.importance(importance_matrix = importance.matrix)



# Extracting the trees: ------------

xgb.dump(bstDMatrix, 
         with_stats = TRUE)

# todo: how to read this? 

# plotting: 
xgb.plot.tree(model = bstDMatrix)


























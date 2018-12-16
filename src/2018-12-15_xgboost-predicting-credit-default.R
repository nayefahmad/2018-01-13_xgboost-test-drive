

#*****************************************************
# USING XGBOOST TO PREDICT CREDIT DEFUALT 
# 2018-12-15
# Nayef 
#*****************************************************


library(xgboost)
library(tidyverse)
library(caret)
library(pROC)
library(ISLR)
library(Matrix)

help(package = "xgboost")


# 1) prep the data: --------------------
data("Default")

df1.default <- Default
str(df1.default)


# Create training data: 
set.seed(1)
train_index <- createDataPartition(df1.default$default, 
                                   p = 0.8, 
                                   list = FALSE, 
                                   times = 1)

df2.train <- df1.default[train_index, ]
df3.test <- df1.default[-train_index, ]
# nrow(df2.train) + nrow(df3.test)

# sparse matrix of predictor values: 
m1.sparse.train <- sparse.model.matrix(default ~ ., 
                                 data = df2.train)[,-1]
head(m1.sparse.train)

# vector of response values: 
train.target <- df2.train$default == "Yes"
# table(train.target)


# put predictor and response together: 
m2.train.dmatrix <- xgb.DMatrix(data = m1.sparse.train, 
                                label = as.numeric(train.target))


# now get the test data ready: 
m3.sparse.test <- sparse.model.matrix(default ~ ., 
                                      data = df3.test)[,-1]

head(m3.sparse.test)

# vector of response values: 
test.target <- df3.test$default == "Yes"
table(test.target)



# 2) Create xgboost model: -----------------------
mod1 <- xgboost(data = m2.train.dmatrix, 
                max_depth = 4, 
                eta = 1, 
                nrounds = 3,  
                objective = "binary:logistic", 
                verbose = TRUE)

# for this problem, looks like train-error decreases
# monotonically with increasing "nrounds" parameter. 
# Maybe because this is simulated data? 

# 3) predictions on the training data: ----------
train.pred <- predict(mod1, 
                      m1.sparse.train)

# use threshold of 0.5 to convert to factor: 
train.pred <- as.numeric(train.pred > 0.5)

table(train.pred)  # predicted responses on training data 
table(train.target)  # actual responses on training data 

# 4) Confusion matrix on training data: -------
confusionMatrix(factor(train.pred), 
                factor(as.numeric(train.target)))

# in the xgboost() call, setting the nrounds parameter too
# high will cause overfitting, meaning the results on the
# train data will not be representative at all of
# performance on actual test data.


# 5) predict on the test set: --------
test.pred <- predict(mod1,
                     m3.sparse.test)

# use threshold of 0.5 to convert to factor: 
test.pred <- as.numeric(test.pred > 0.5)

table(test.pred)  # predicted responses on test data 
table(test.target)  # actual responses on test data 

# 6) Confusion matrix on training data: -------
confusionMatrix(factor(test.pred), 
                factor(as.numeric(test.target)))

# Sensitivity is still good, but specificity is quite bad: 
# 0.2727
# todo: how to increase the specificity?? 


# 7) roc curve: -----------
roc1 <- roc(test.target, 
            test.pred, 
            algorithm = 2)

plot(roc1)
auc(roc1)  # Area under the curve: 0.6312. This is not great. 



#********************************************************
# 8) Alternative model development using cross-validation: ------------
# Note: first do CV, then do one final prediction on the 
# test data
# reference: https://stats.stackexchange.com/questions/152907/how-do-you-use-test-data-set-after-cross-validation 


















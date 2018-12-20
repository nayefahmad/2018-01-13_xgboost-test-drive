

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

# rm(list = ls())

# 1) prep the data: --------------------
data("Default")

df1.default <- Default
str(df1.default)


# > 1.1) Create training data: -------------
set.seed(1)
train_index <- createDataPartition(df1.default$default, 
                                   p = 0.8, 
                                   list = FALSE, 
                                   times = 1)

df2.train <- df1.default[train_index, ]
df3.test <- df1.default[-train_index, ]
# nrow(df2.train) + nrow(df3.test)

# >> 1.1.1) checking whether xgboost can handle NA values in predictors: ------
# df2.train[1,3] <- NA
# df2.train[2,3] <- "NULL"
# head(df2.train)

# sparse matrix of predictor values: 
m1.train.predictors <- sparse.model.matrix(default ~ ., 
                                 data = df2.train)[,-1]
head(m1.train.predictors)
# looks like sparse.mode.matrix( ) removed entire row with
# the NA value
# todo: can we prevent this? 


# > 1.2) training data: vector of response values: ------
train.target <- as.numeric(df2.train$default == "Yes") 
# table(train.target)


# put predictor and response together: 
m2.train.dmatrix <- xgb.DMatrix(data = m1.train.predictors, 
                                label = train.target)


# > 1.3) test set data ready: 
m3.test.predictors <- sparse.model.matrix(default ~ ., 
                                      data = df3.test)[,-1]

head(m3.test.predictors)

# vector of response values: 
test.target <- as.numeric(df3.test$default == "Yes") 
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


# 2.1) cross-validation to estimate test set error: -------
# Note: first do CV, then do one final prediction on the 
# test data
# reference: https://stats.stackexchange.com/questions/152907/how-do-you-use-test-data-set-after-cross-validation 

mod.cv <- xgb.cv(data = m2.train.dmatrix, 
                 max_depth = 4, 
                 eta = 1, 
                 nrounds = 3,  
                 objective = "binary:logistic",
                 nfold = 5, 
                 metrics = "auc", 
                 verbose = TRUE)

# since training and validation AUC (denoted "test-auc" here) are close, looks
# like we're not overfitting to training data


# in the xgboost() call, setting the max_depth parameter too
# high will cause overfitting, meaning the results on the
# train data will not be representative at all of
# performance on actual test data.



# 3) ROC curve on train data: ----------
train.pred <- predict(mod1, m1.train.predictors)

# ?roc

roc1 <- roc(train.target,  # actual response values  
            train.pred, 
            algorithm = 1)  # predicted response values 

plot(roc1, col = "blue", 
     main = paste0("training set AUC = ",
                   auc(roc1) %>% round(2)))
auc(roc1)  # Area under the curve: 0.9609


# > 3.1 extract thresholds from the ROC curve: ------
# ?coords
roc1.coords <- coords(roc = roc1, x = "all")  # todo: what's happening??
dim(roc1.coords)  # 3 57
roc1.coords

# let's say our goal is really high sensitivity. with decent
# specificity
roc1.coords[, roc1.coords["sensitivity", ] >= .90]

# threshold of 0.03 seems good. 

# use threshold of 0.05 to convert prob to classification: 
train.pred.class <- as.numeric(train.pred > 0.03)
table(train.pred.class)


# just checking, what if we used default threshold of 0.5? 
coords(roc = roc1, 
       x = 0.5)




# 4) Confusion matrix on training data: -------
table(train.pred.class)  # predicted responses on training data 
table(train.target)  # actual responses on training data 


confusionMatrix(factor(train.pred.class), 
                factor(as.numeric(train.target)))



# 5) interpreting the model built on training data: -----
df4.mod1.imp <- xgb.importance(feature_names = colnames(m1.train.predictors), 
                               model = mod1)

head(df4.mod1.imp, 15)

xgb.plot.importance(head(df4.mod1.imp, 15))


# visualise one tree: 
xgb.plot.tree(feature_names = colnames(m1.train.predictors), 
              model = mod1, 
              trees = 0)  # uses zero-based counting 

# visualize a lot of trees: 
# todo: doesn't work

# list1.trees <- lapply(1:9, 
#                       xgb.plot.tree, 
#                       feature_names = colnames(m1.train.predictors),
#                       model = mod1)
# list1.trees

# export one graph 
# todo: doesn't work
# export_graph(list1.trees[[1]],  
#              here::here("results",
#                         "output from src",
#                         "2018-12-20_xgboost-trees.pdf"))





# 6) Confusion matrix on test data: -------
test.pred <- predict(mod1,m3.test.predictors)

# use threshold of 0.03 to convert to factor: 
test.pred.class <- as.numeric(test.pred > 0.03)

table(test.pred.class)  # predicted responses on test data 
table(test.target)  # actual responses on test data 


confusionMatrix(factor(test.pred.class), 
                factor(as.numeric(test.target)))

# out of 66 ppl in the test dataset who defaulted, we
# correctly identified 58 of them. We miscalssified 8/66 = 12%

# however, on the flip side, out of 1933 non-defaulters, we
# misclassified 354/1933 = 18%


# 7) roc curve on test data: -----------
roc2 <- roc(test.target, 
            test.pred, 
            algorithm = 1)

plot(roc2, 
     col = "blue", 
     main = paste0("test set auc = ", 
                   auc(roc2) %>% round(2)))
auc(roc2)  # Area under the curve: 0.9254 This is great. 
















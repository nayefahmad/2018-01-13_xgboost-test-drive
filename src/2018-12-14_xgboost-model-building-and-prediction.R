

#*****************************************************
# USING XGBOOST TO BUILD A MODEL AND PREDICT STUFF 
# 2018-12-15
# Nayef 
#*****************************************************


library(xgboost)
library(tidyverse)

data("agaricus.train")
data("agaricus.test")

train <- agaricus.train
test <- agaricus.test

# data is in list form: 
str(train, max.level = 1)

# extract first element of list: 
train$data %>% str
# class 'dgCMatrix' [package "Matrix"] with 6 slots
# dgCMatrix is the “standard” class for sparse numeric
# matrices in the Matrix package.



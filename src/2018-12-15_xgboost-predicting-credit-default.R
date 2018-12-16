

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

help(package = "xgboost")


# 1) prep the data: --------------------
data("Default")

df1.default <- Default

str(df1.default)











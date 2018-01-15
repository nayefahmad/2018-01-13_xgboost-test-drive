

#******************************
# TESTING XGBOOST PACKAGE 
#******************************

help(package = "xgboost")

require(xgboost)
require(Matrix)
require(data.table)
if (!require('vcd')) install.packages('vcd')

# TODO: -----------------------
# > data.table syntax 

#******************************




# load arthritis data: 
data(Arthritis)
df <- data.table(Arthritis, keep.rownames = FALSE)

head(df)
str(df)


# experiment1: group ages into buckets of 10: -----
head(df[ , AgeDiscret := as.factor(round(Age/10, 0))])

# note: above we're using data.table syntax. Not yet completely 
#     sure how that works. 

# experiment2: arbitrarily group ages into 2 buckets: 
head(df[ , AgeCat := as.factor(ifelse(Age > 30, 
                                      "Old", 
                                      "Young"))])

# note that these new cols are highly correlated with Age. 
# This is not a problem for a decision tree (unlike a GLM)


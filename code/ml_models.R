# This is a R script for analyzing the facility life-cycle cost data
# Machine learning models for single/multi target regression:   
# Developed by Xinghua Gao @ Georgia Tech
# Email: gaoxh@gatech.edu
# March 2019

# Load libraries
library(ggplot2)
library(stringr)
library(hydroGOF)
library(Metrics)
library(class)
library(caret)
library(pls)
library(FNN)
library(rpart)
library(boot)
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)
library(tensorflow)
library(e1071)
library(gmodels) 
library(psych)
library(randomForest)
library(MultivariateRandomForest)

################## Parameter setting ################## 
# how many iterations? (100 by default)
loop = 3

# set a threshold for the buildng age; 
# the buildings older than this age won't be used in
# the model development
age = 100

# set the model to train, 0 means not train, 1 means train
MLR = 1 # Linear regression
KNN = 1 # KNN
tree = 1 # random forest
SVM = 1 # SVM
MLP = 0 # multilayer perceptron
tree_multi = 1 # multi-output random forest
MLP_multi = 0 # multilayer perceptron (multi-target)

# number of descriptive attributes
num_des = 16

# the k value of the KNN model
knn_k = 3

# epochs of the multilayer perceptron model training
num_epo = 100

# batch size of the multilayer perceptron model training
num_batch = 90

# the validation split of the multilayer perceptron model training
# From 0.01 to 0.99, the percent of validation set.
val_split = 0.02

# descriptive attributes for initial cost prediction
attri_initial <- initial ~ gsf + age + floor + 
  BLDG_SVC + CIRC +MECH + CLS_FAC + LAB_FAC + 
  OFF_FAC + STDY_FAC + SPEC_USE + GEN_USE +
  SUPP_FAC  + HLTH_FAC + RES_FAC + other

# descriptive attributes for utility cost prediction
attri_utility <- utility ~ gsf + age + floor + 
  BLDG_SVC + CIRC +MECH + CLS_FAC + LAB_FAC + 
  OFF_FAC + STDY_FAC + SPEC_USE + GEN_USE +
  SUPP_FAC  + HLTH_FAC + RES_FAC + other

# descriptive attributes for O&M cost prediction
attri_om <- om ~ gsf + age + floor + 
  BLDG_SVC + CIRC +MECH + CLS_FAC + LAB_FAC + 
  OFF_FAC + STDY_FAC + SPEC_USE + GEN_USE +
  SUPP_FAC  + HLTH_FAC + RES_FAC + other

################## END Parameter setting ################## 


################## Data importing and processing ##################

# Load raw data
data <- read.csv("train.csv", header = TRUE, check.names = TRUE)

# Change column name to solve the weird column header name issue
names(data)[1] <- "id"

# don't need the predictor "year" becasue already have the predictor "age"
data <- data[,-6]

# Remove the instance with the largest utility cost: 189 Substation Control House
# It is a control house, its utility data are representing many other buildings
data <- data[-which.max(data$utility),]

# Remove the instance with the largest om cost: 73 McCamish Pavilion
# It is recently renovated in 2012 and the removation costs are recorded in the AiM system
data <- data[-which.max(data$om),]

# Remove the O'Keefe, Daniel C.building, which O&M cost is abnormal
# It just had a major renovation recently and the cost is recorded as maintenance cost
data <- data[-which(data$id==33),]

# The utility consumption of building 138 is abnormal
data <- data[-which(data$id==138),]

# convert the predictors in Factor format to numeric
data %<>% mutate_if(is.factor, as.numeric)

# Create the data frame for the cost per quare foot
data.persf <- data
data.persf$initial <- data.persf$initial*1000/data.persf$gsf
data.persf$utility <- data.persf$utility*1000/data.persf$gsf
data.persf$om <- data.persf$om*1000/data.persf$gsf

# Remove the outliers of utility cost, based on the cost per SF
data <- data[-which(data.persf$utility>=400),]
data.persf <- data.persf[-which(data.persf$utility>=400),]

# Remove the outliers of O&M cost, based on the cost per SF
data <- data[-which(data.persf$om>=1000),]
data.persf <- data.persf[-which(data.persf$om>=1000),]


# Remove the instances with age older than the threshold
data <- data[!(data$age >age),]
data.persf <- data.persf[!(data.persf$age >age),]

################## END Data importing and processing ##################


################## Define the result data array ##################
# results <- data.frame("method" = c("MLR(single)", "KNN(single)",
#                                    "tree(single)", "SVM(single)", "MLP(single)",
#                                    "tree(multi)", "MLP(multi)"),
#                       "initial" = 1:7, "utility" = 1:7, "om" = 1:7)

# record each iteration
result_table <- array(0,dim=c(7,3,loop))

# record the mean of all iteration
result <- array(0,dim=c(7,3))

# The counter for valid loops
# Sometimes the loop may be skipped
counter = 0

################## END Define the result data frame ##################


################## Loop starts ################## 
for (i in 1:loop){

################## Define the training set and test set ##################

  
# set the random seed if needed
# set.seed(123)
  
# data partition
ind <- sample(2, nrow(data), replace =T, prob = c(.8,.2))

training <- data.frame(data[ind==1,-c(1:5,7:9,11,13)])
test <- data.frame(data[ind==2,-c(1:5,7:9,11,13)])

trainingtarget <- data.frame(data[ind==1, c(3:5)])
testtarget <- data.frame(data[ind==2, c(3:5)])

# normalize

m <- colMeans(training)
s <- apply(training,2,sd)

# some times the m and s have zero elements, which make the training and test set have Nah
# in this case, skip to the next loop
if (any(m == 0)||any(s == 0)) {
  print(paste0("round ", i, ", the loop is skipped"))
  next
}

training <- data.frame(scale (training, center = m, scale =s))
test <- data.frame(scale (test, center = m, scale =s))

# normalize targets to test 
m2 <- colMeans(trainingtarget) # here do use the mean and SD of the trainingtarget
s2 <- apply(trainingtarget,2,sd)
trainingtarget <- data.frame(scale (trainingtarget, center = m2, scale =s2))
testtarget <- data.frame(scale (testtarget, center = m2, scale =s2))

# individual targets for some R packages
training.initial <- trainingtarget[c(1)]
training.utility <- trainingtarget[c(2)]
training.om <- trainingtarget[c(3)]

test.initial <- testtarget[c(1)]
test.utility <- testtarget[c(2)]
test.om <- testtarget[c(3)]

# merge the training and test set for some R packages
training.merge <- data.frame(trainingtarget,training)
test.merge <- data.frame(testtarget,test)

################## END Define the training set and test set ##################


################## Model development and validation ##################

### Multilinear regression model (single target) ###

if (MLR == 1){

# MRL for initial cost
MLR_S_initial <- lm(attri_initial, data = training.merge)

# MRL for utility cost
MLR_S_utility <- lm(attri_utility, data = training.merge)

# MRL for O&M cost
MLR_S_om <- lm(attri_om, data = training.merge)


# Predictions using the developed linear models
pred_MLRSI <- predict (MLR_S_initial, test.merge) # initial
pred_MLRSU <- predict (MLR_S_initial, test.merge) # utility
pred_MLRSO <- predict (MLR_S_initial, test.merge) # om

# MLR validation

# Root Mean Squared Error (RMSE)
# RMSE = rmse(predY,test.merge$initial)

# Mean squared error (MSE)
#MSE = mse(predY,test.merge$initial)

# Relative absolute error (RAE)
#RAE = rae(test.merge$initial, predY)

# Mean absolute error (MAE)
MAE_MLRSI = mae(test.merge$initial,pred_MLRSI)
MAE_MLRSU = mae(test.merge$utility,pred_MLRSU)
MAE_MLRSO = mae(test.merge$om,pred_MLRSO)

# record the results
result_table[1,1,i] = MAE_MLRSI;
result_table[1,2,i] = MAE_MLRSU;
result_table[1,3,i] = MAE_MLRSO;

}

### END Multilinear regression model (single target) ###

### KNN regression model (single target) ###

if (KNN == 1){

# predictions based on the test set
# the experimens indicated that k = 3 yeilds best results
pred_KNNSI <- knn.reg(training, test, training.initial, k = knn_k)
pred_KNNSU <- knn.reg(training, test, training.utility, k = knn_k)
pred_KNNSO <- knn.reg(training, test, training.om, k = knn_k)

# Mean absolute error (MAE)
MAE_KNNSI = mae(testtarget$initial,pred_KNNSI$pred)
MAE_KNNSU = mae(testtarget$utility,pred_KNNSU$pred)
MAE_KNNSO = mae(testtarget$om,pred_KNNSO$pred)

# record the results
result_table[2,1,i] = MAE_KNNSI;
result_table[2,2,i] = MAE_KNNSU;
result_table[2,3,i] = MAE_KNNSO;

}

### END KNN regression model (single target) ###


### Regression Tree model (single target) ###

if (tree == 1){

# random forest for initial cost
forest_SI <- randomForest(attri_initial,
                          data = training.merge)

# random forest for utility cost
forest_SU <- randomForest(attri_utility,
                          data = training.merge)

# random forest for O&M cost
forest_SO <- randomForest(attri_om,
                          data = training.merge)
  
  
# predictions based on the test set
pred_treeSI <- predict(forest_SI, test)
pred_treeSU <- predict(forest_SU, test)
pred_treeSO <- predict(forest_SO, test)

# Mean absolute error (MAE)
MAE_treeSI <- mae(testtarget$initial,pred_treeSI)
MAE_treeSU <- mae(testtarget$utility,pred_treeSU)
MAE_treeSO <- mae(testtarget$om,pred_treeSO)

# record the results
result_table[3,1,i] = MAE_treeSI;
result_table[3,2,i] = MAE_treeSU;
result_table[3,3,i] = MAE_treeSO;

}

### END Regression Tree model (single target) ###


### SVM Regression model (single target) ###

if (SVM == 1){

# SVM regression for initial cost
SVM_S_initial <- svm(attri_initial,
                     data = training.merge)

# SVM regression for utility cost
SVM_S_utility <- svm(attri_utility,
                     data = training.merge)

# SVM regression for O&M cost
SVM_S_om <- svm(attri_om,
                     data = training.merge)

# predictions based on the test set
pred_SVMSI <- predict(SVM_S_initial, test)
pred_SVMSU <- predict(SVM_S_utility, test)
pred_SVMSO <- predict(SVM_S_om, test)

# Mean absolute error (MAE)
MAE_SVMSI <- mae(testtarget$initial,pred_SVMSI)
MAE_SVMSU <- mae(testtarget$utility,pred_SVMSU)
MAE_SVMSO <- mae(testtarget$om,pred_SVMSO)

# record the results
result_table[4,1,i] = MAE_SVMSI;
result_table[4,2,i] = MAE_SVMSU;
result_table[4,3,i] = MAE_SVMSO;

}

### END SVM Regression model (single target) ###


# converge the data into matrix
training_m <- as.matrix(training)
trainingtarget_m <- as.matrix(trainingtarget)
training.initial_m <- as.matrix(training.initial)
training.utility_m <- as.matrix(training.utility)
training.om_m <- as.matrix(training.om)
test_m <- as.matrix(test)
testtarget_m <- as.matrix(testtarget)
test.initial_m <- as.matrix(test.initial)
test.utility_m <- as.matrix(test.utility)
test.om_m <- as.matrix(test.om)


### Multilayer perceptron model (single target) ###

if (MLP == 1){

# create MLP model (empty)
model_SI <- keras_model_sequential()
model_SI  %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = c(num_des)) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'relu') %>%
  layer_dense(units = 1)

# compile
model_SI  %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

model_SU <- model_SI
model_SO <- model_SI

# fit the MLP model for initial cost prediction
MLP_SI <- model_SI  %>%
  fit(training_m,
      training.initial_m,
      epochs = num_epo,
      batch_size = num_batch,
      validation_split = val_split)

# fit the MLP model for utility cost prediction
MLP_SU <- model_SU %>%
  fit(training_m,
      training.utility_m,
      epochs = num_epo,
      batch_size = num_batch,
      validation_split = val_split)

# fit the MLP model for O&M cost prediction
MLP_SO <- model_SO %>%
  fit(training_m,
      training.om_m,
      epochs = num_epo,
      batch_size = num_batch,
      validation_split = val_split)

# predictions based on the test set
pred_MLP_SI <- model_SI %>% predict(test_m)
pred_MLP_SU <- model_SU %>% predict(test_m)
pred_MLP_SO <- model_SO %>% predict(test_m)

# Mean absolute error (MAE)
MAE_MLP_SI <- mae(testtarget$initial,pred_MLP_SI)
MAE_MLP_SU <- mae(testtarget$utility,pred_MLP_SU)
MAE_MLP_SO <- mae(testtarget$om,pred_MLP_SO)

# record the results
result_table[5,1,i] = MAE_MLP_SI;
result_table[5,2,i] = MAE_MLP_SU;
result_table[5,3,i] = MAE_MLP_SO;

}

### END Multilayer perceptron model (single target) ###


### Regression Tree model (multi target) ###

if (tree_multi == 1){

# Multivariate Random Forest model
# build_forest_predict(trainX, trainY, n_tree, m_feature, min_leaf, testX)
forest_M <- build_forest_predict(training_m, trainingtarget_m, 100, 10, 40, test_m)

# Evaluation
MAE_forest_MI <- mae(testtarget$initial,forest_M[,1])
MAE_forest_MU <- mae(testtarget$utility,forest_M[,2])
MAE_forest_MO <- mae(testtarget$om,forest_M[,3])

# record the results
result_table[6,1,i] = MAE_forest_MI;
result_table[6,2,i] = MAE_forest_MU;
result_table[6,3,i] = MAE_forest_MO;

}

### END Regression Tree model (multi target) ###


### Multilayer perceptron model (multi target) ###

if (MLP_multi == 1){

# create MLP model (empty)
model_M <- keras_model_sequential()
model_M  %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = c(num_des)) %>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 5, activation = 'relu') %>%
  layer_dense(units = 3)

# compile
model_M  %>% compile(loss = 'mse',
                      optimizer = 'rmsprop',
                      metrics = 'mae')

# fit the model
MLP_SI <- model_M  %>%
  fit(training_m,
      trainingtarget_m,
      epochs = num_epo,
      batch_size = num_batch,
      validation_split = val_split)

# predictions based on the test set
pred_MLP_M <- model_M %>% predict(test_m)

# Mean absolute error (MAE)
MAE_MLP_MI <- mae(testtarget$initial,pred_MLP_M[,1])
MAE_MLP_MU <- mae(testtarget$utility,pred_MLP_M[,2])
MAE_MLP_MO <- mae(testtarget$om,pred_MLP_M[,3])

# record the results
result_table[7,1,i] = MAE_MLP_MI;
result_table[7,2,i] = MAE_MLP_MU;
result_table[7,3,i] = MAE_MLP_MO;

}

### END Multilayer perceptron model (multi target) ###


################## Loop ends ################## 
print(paste0("round ", i, " finished"))

counter = counter + 1

}

################## Results Output ##################

for (i in 1:7){
  for (j in 1:3){
    for (k in 1:counter){
        result[i,j] = result[i,j] + result_table[i,j,k]
    }
    result[i,j] = result[i,j]/counter
  }
}

print(result)

################## END Results Output ##################
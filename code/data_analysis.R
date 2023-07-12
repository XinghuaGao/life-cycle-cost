# This is a R script for analyzing the facility life-cycle cost data
# Basic Data Analysis
# Developed by Xinghua Gao @ Georgia Tech
# Email: gaoxh@gatech.edu
# March 2019

# Load libraries
library(ggplot2)
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)
library(tensorflow)

################## Data importing and processing ##################

# Load raw data
data <- read.csv("train.csv", header = TRUE, check.names = TRUE)

# Change column name to solve the weird column header name issue
names(data)[1] <- "id"

# Don't need the predictor "year" because already have the predictor "age"
data <- data[,-6]

# Initial attempt to convert factors to numeric was commented out
# data %<>% mutate_if(is.factor, as.numeric)

# Removing outliers and abnormal instances
data <- data[-which.max(data$utility),]   # Remove instance with largest utility cost
data <- data[-which.max(data$om),]        # Remove instance with largest om cost
data <- data[-which(data$id==138),]       # Remove building 138 due to abnormal utility consumption
data <- data[-which(data$id==33),]        # Remove O'Keefe, Daniel C. building due to abnormal O&M cost

# Create a data frame for the cost per square foot
data.persf <- data
data.persf$initial <- data.persf$initial*1000/data.persf$gsf
data.persf$utility <- data.persf$utility*1000/data.persf$gsf
data.persf$om <- data.persf$om*1000/data.persf$gsf

# Removing outliers based on cost per square foot
data <- data[-which(data.persf$utility>=400),]
data.persf <- data.persf[-which(data.persf$utility>=400),]

data <- data[-which(data.persf$om>=1000),]
data.persf <- data.persf[-which(data.persf$om>=1000),]

# Preparing data for checking the correlations of main parameters
data_2 <- data[, which(names(data) %in% c("initial","utility","om","gsf","floor","age"))]
data.persf_2 <- data.persf[, which(names(data.persf) %in% c("initial","utility","om","gsf","floor","age"))]

# Calculating rates
rate_iu <- data$utility/data$initial
rate_iom <- data$om/data$initial
rate_iu_ps <- data.persf$utility/data.persf$initial
rate_iom_ps <- data.persf$om/data.persf$initial

################## Plot ##################

# Plotting correlations
pairs(data_2)
cor <- cor(data[,-c(1:2,7:9,11,13)])

pairs(data.persf_2)
cor.persf <- cor(data.persf[,-c(1:2,7:9,11,13)])

# Plotting histograms
hist(data.persf$initial,breaks=100)
hist(data.persf$utility,breaks=100)
hist(data.persf$om,breaks=100)

hist(rate_iu, breaks=100)
hist(rate_iom, breaks=100)

# Find the building with largest rate
max_iu <- data$id[which.max(rate_iu)]
max_iom <- data$id[which.max(rate_iom)]

# Plot the count of buildings by owner and cost per SF
ggplot(data.persf[data.persf$owner != "n/a",], aes(x = initial)) +
  facet_wrap(~owner) +
  geom_histogram(binwidth = 15) +
  ggtitle("The count of buildings by owner and initial cost") +
  xlab("Initial cost per SF") +
  ylab("The number of buildings")

ggplot(data.persf[data.persf$owner != "n/a",], aes(x = utility)) +
  facet_wrap(~owner) +
  geom_histogram(binwidth = 15) +
  ggtitle("The count of buildings by owner and utility cost") +
  xlab("Utility cost per SF") +
  ylab("The number of buildings")

ggplot(data.persf[data.persf$owner != "n/a",], aes(x = om)) +
  facet_wrap(~owner) +
  geom_histogram(binwidth = 15) +
  ggtitle("The count of buildings by owner and O&M cost") +
  xlab("O&M cost per SF") +
  ylab("The number of buildings")

################## Write file ##################

# Write correlation tables to csv files
write.table(cor, file = "Correlation.csv", sep = ",", col.names = NA, qmethod = "double")
write.table(cor.persf, file = "Correlation_persf.csv", sep = ",", col.names = NA, qmethod = "double")
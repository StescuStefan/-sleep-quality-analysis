###############################################################################
# Sleep Quality and Lifestyle Factors Analysis
# 
# Author: Stefan Stescu
# Date: January 2025
#
# This script analyzes the relationships between sleep quality, stress levels,
# and various lifestyle factors using statistical methods in R.
###############################################################################

#------------------------------------------------------------------------------
# Load Required Libraries
#------------------------------------------------------------------------------
library(tidyverse)  # Data manipulation and visualization
library(dplyr)      # Data manipulation
library(tidyr)      # Data cleaning
library(stringr)    # String manipulation
library(boot)       # Bootstrap resampling
library(psych)      # Psychological statistics
library(stats)      # Statistical functions

#------------------------------------------------------------------------------
# Data Import and Processing
#------------------------------------------------------------------------------
# Import the dataset from the CSV file
sleep <- read.csv("Sleep_health_and_lifestyle_dataset.csv")

# Create a filtered subset based on specific conditions
sleep_quality_check <- subset(
  sleep, 
  (Sleep.Duration >= 5.0 & Sleep.Duration <= 7.5) & 
    (Stress.Level >= 2 & Stress.Level <= 9) & 
    (Quality.of.Sleep >= 3 & Quality.of.Sleep <= 9), 
  select = c(Person.ID, Gender, Age, Occupation, Sleep.Duration, 
             Stress.Level, Quality.of.Sleep, Physical.Activity.Level, 
             BMI.Category, Heart.Rate, Sleep.Disorder)
)

# Optional: Remove original dataframe to free memory
# rm(sleep)

# Check for missing values
colSums(is.na(sleep_quality_check))

# Save the filtered dataset to a CSV file
write.csv(sleep_quality_check, "sleep_quality_check.csv", row.names = FALSE)

# Convert Sleep.Disorder to a factor
sleep_quality_check$Sleep.Disorder <- as.factor(sleep_quality_check$Sleep.Disorder)

# Create a categorical variable for stress levels
sleep_quality_check$sleep_disorder_stress <- cut(
  sleep_quality_check$Stress.Level, 
  c(0, 3, 5, 9),
  c("Low", "Moderate", "High")
)

#------------------------------------------------------------------------------
# Exploratory Data Analysis
#------------------------------------------------------------------------------
# Display basic information about the dataset
dim(sleep_quality_check)      # Dimensions
str(sleep_quality_check)      # Structure
names(sleep_quality_check)    # Variable names

# Check factor levels
levels(sleep_quality_check$sleep_disorder_stress)
levels(sleep_quality_check$Sleep.Disorder)

# Create a numerical subset for analysis
data_sep_num <- subset(
  sleep_quality_check, 
  select = c(Age, Sleep.Duration, Stress.Level, Quality.of.Sleep, 
             Physical.Activity.Level, Heart.Rate)
)

# Descriptive statistics for numerical variables
describe(data_sep_num)
summary(data_sep_num)

# Summary statistics for individual variables
summary(sleep_quality_check$Age)
summary(sleep_quality_check$Sleep.Duration)
summary(sleep_quality_check$Stress.Level)
summary(sleep_quality_check$Quality.of.Sleep)
summary(sleep_quality_check$Physical.Activity.Level)
summary(sleep_quality_check$Heart.Rate)

# Group statistics
describeBy(
  sleep_quality_check$Sleep.Duration,
  group = sleep_quality_check$sleep_disorder_stress, 
  digits = 4
)

describeBy(
  data_sep_num, 
  group = sleep_quality_check$Sleep.Disorder, 
  digits = 4
)

# Calculate means by groups
tapply(
  sleep_quality_check$Quality.of.Sleep, 
  sleep_quality_check$sleep_disorder_stress, 
  mean
)

aggregate(
  Quality.of.Sleep ~ Sleep.Disorder, 
  sleep_quality_check, 
  mean
)

#------------------------------------------------------------------------------
# Data Visualization
#------------------------------------------------------------------------------
# Histograms for numerical variables
hist(
  sleep_quality_check$Age, 
  xlab = "Age of the people involved in the analysis",
  main = "Distribution of Age",
  col = "lightblue",
  border = "white"
)

hist(
  sleep_quality_check$Sleep.Duration, 
  xlab = "Sleep Duration per night",
  main = "Distribution of Sleep Duration",
  col = "lightblue",
  border = "white"
)

hist(
  sleep_quality_check$Stress.Level, 
  xlab = "Stress Level",
  main = "Distribution of Stress Level",
  col = "lightblue",
  border = "white"
)

hist(
  sleep_quality_check$Quality.of.Sleep, 
  xlab = "Quality of Sleep",
  main = "Distribution of Sleep Quality",
  col = "lightblue",
  border = "white"
)

hist(
  sleep_quality_check$Physical.Activity.Level, 
  xlab = "Physical Activity Level",
  main = "Distribution of Physical Activity",
  col = "lightblue",
  border = "white"
)

hist(
  sleep_quality_check$Heart.Rate, 
  xlab = "Heart Rate",
  main = "Distribution of Heart Rate",
  col = "lightblue",
  border = "white"
)

# Visualizing categorical variables
plot(
  sleep_quality_check$Sleep.Disorder, 
  xlab = "Sleep Disorder",
  main = "Sleep Disorders Distribution",
  col = c("lightblue", "green", "brown")
)

plot(
  sleep_quality_check$sleep_disorder_stress, 
  xlab = "The level of stress",
  main = "Stress Level Categories",
  col = c("green", "orange", "red")
)

# Boxplots to identify outliers
boxplot(
  sleep_quality_check$Age, 
  main = "Boxplot of Age",
  xlab = "Age of the people involved",
  col = "lightgreen"
)
  
boxplot(
  sleep_quality_check$Sleep.Duration, 
  main = "Boxplot of Sleep Duration",
  xlab = "Sleep Duration",
  col = "lightgreen"
)

boxplot(
  sleep_quality_check$Stress.Level, 
  main = "Boxplot of Stress Level",
  xlab = "Stress Level",
  col = "lightgreen"
)

boxplot(
  sleep_quality_check$Quality.of.Sleep, 
  main = "Boxplot of Quality of Sleep",
  xlab = "Quality of Sleep",
  col = "lightgreen"
)

boxplot(
  sleep_quality_check$Physical.Activity.Level, 
  main = "Boxplot of Physical Activity Level",
  xlab = "Physical Activity Level",
  col = "lightgreen"
)

boxplot(
  sleep_quality_check$Heart.Rate, 
  main = "Boxplot of Heart Rate",
  xlab = "Heart Rate",
  col = "lightgreen"
)

#------------------------------------------------------------------------------
# Cross-tabulation Analysis
#------------------------------------------------------------------------------
# Create a cross-tabulation table
tabelarea_datelor <- table(
  sleep_quality_check$Sleep.Disorder,
  sleep_quality_check$sleep_disorder_stress
)
tabelarea_datelor

# Calculate relative frequencies
prop.table(tabelarea_datelor)

# Row-conditional relative frequencies
prop.table(tabelarea_datelor, 1)

# Column-conditional relative frequencies
prop.table(tabelarea_datelor, 2)

# Marginal frequencies
addmargins(prop.table(tabelarea_datelor))

# Summary of the table
summary(tabelarea_datelor)

#------------------------------------------------------------------------------
# Chi-Square Tests
#------------------------------------------------------------------------------
# Chi-square test for sleep_disorder_stress distribution
chisq.test(table(sleep_quality_check$sleep_disorder_stress))

# Chi-square test with specified theoretical distribution
chisq.test(
  table(sleep_quality_check$sleep_disorder_stress), 
  p = c(0.1, 0.3, 0.6)
)

# Chi-square test for Sleep.Disorder distribution
chisq.test(table(sleep_quality_check$Sleep.Disorder))

# Chi-square test with specified theoretical distribution
chisq.test(
  table(sleep_quality_check$Sleep.Disorder), 
  p = c(0.1, 0.2, 0.7)
)

#------------------------------------------------------------------------------
# Correlation Analysis
#------------------------------------------------------------------------------
# Covariance matrix
cov(data_sep_num)

# Pearson correlation matrix
cor(data_sep_num)

# Spearman correlation matrix
cor(data_sep_num, method = 'spearman')

# Correlation tests for Stress Level with other variables
cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Age)
cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Sleep.Duration)
cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Quality.of.Sleep)
cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Physical.Activity.Level)
cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Heart.Rate)

#------------------------------------------------------------------------------
# Regression Analysis
#------------------------------------------------------------------------------
# Simple linear regression model
regresie_lin_simpla <- lm(Stress.Level ~ Quality.of.Sleep, sleep_quality_check)
summary(regresie_lin_simpla)

# Multiple linear regression model
regresie_lin_multipla <- lm(Stress.Level ~ Quality.of.Sleep + Age, sleep_quality_check)
summary(regresie_lin_multipla)

# Non-linear regression model (quadratic model)
regresie_neliniara <- lm(
  Stress.Level ~ Quality.of.Sleep + I(Quality.of.Sleep^2), 
  sleep_quality_check
)
summary(regresie_neliniara)

# Compare regression models using ANOVA
anova(regresie_lin_simpla, regresie_lin_multipla)

#------------------------------------------------------------------------------
# Hypothesis Testing
#------------------------------------------------------------------------------
# One-sample t-test (comparing Stress.Level to a fixed value)
t.test(sleep_quality_check$Stress.Level, mu = 0)

# One-sided t-test
t.test(
  sleep_quality_check$Stress.Level, 
  mu = 5, 
  alternative = "greater"
)

# Test for equal variances between Sleep Disorder groups
bartlett.test(
  Stress.Level ~ Sleep.Disorder, 
  sleep_quality_check,
  Sleep.Disorder %in% c("Insomnia", "Sleep Apnea")
)

# Create subset for Sleep Disorder comparison
subset_sleep_quality_check <- sleep_quality_check[
  sleep_quality_check$Sleep.Disorder %in% c("Insomnia", "Sleep Apnea"), 
]

# Two-sample t-test comparing Stress.Level between Sleep Disorder groups
t.test(Stress.Level ~ Sleep.Disorder, subset_sleep_quality_check)

# ANOVA test for Stress.Level across sleep_disorder_stress groups
objectaov <- aov(Stress.Level ~ sleep_disorder_stress, sleep_quality_check)
anova(objectaov)
coef(objectaov)

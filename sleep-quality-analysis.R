library(tidyverse)
library(dplyr)
library(tidyr)
library(stringr)
library(boot)
library(psych)
library(stats)

#importing the data set from the CSV file (Kaggle)
sleep <- read.csv("Sleep_health_and_lifestyle_dataset.csv")

sleep_quality_check <- subset(sleep, 
                              (Sleep.Duration >= 5.0 & Sleep.Duration <= 7.5) & 
                                (Stress.Level >= 2 & Stress.Level <= 9) & 
                                (Quality.of.Sleep >= 3 & Quality.of.Sleep <= 9), 
                              select = c(Person.ID, Gender, Age, Occupation, Sleep.Duration, Stress.Level, Quality.of.Sleep,
                                         Physical.Activity.Level, BMI.Category, Heart.Rate, Sleep.Disorder))

#removing the data frame if necessary and other variables
rm(sleep)
rm(sleep_quality_check)
rm(data_sep_num)
#colnames(sleep) <-make.names(colnames(sleep))
#view a brief summary of the new subset we've just created
View(sleep_quality_check)

#checking if there are NA values in the data
colSums(is.na(sleep_quality_check))
#renaming the first column
colnames(sleep)[which(colnames(sleep) == "")] <- "Person.ID"
colnames(sleep_quality_check)

# create a new CSV file with only the data that meet the conditions from above
write.csv(sleep_quality_check, "C:/Users/stesc/Documents/Proiect in R/sleep_quality_check.csv", row.names = FALSE)
#checking the classes of the variables
sapply(sleep_quality_check, class)

sleep_quality_check$Sleep.Disorder <- as.factor(sleep_quality_check$Sleep.Disorder)

#changing the data type of variable Gender
#sleep_quality_check$Gender <- as.character(sleep_quality_check$Gender)


#sleep_quality_check$overall_physical_state <- cut(sleep_quality_check$Heart.Rate, 
                                                  #c(0, 60, 100, 200),de
                                                  #c("Low", "Normal", "High"))

#removing the "overall_physical_state" column from the data frame
#sleep_quality_check <- sleep_quality_check[,-which(names(sleep_quality_check) == "overall_physical_state")]

#removing the column "sleep_disorder_stress"
sleep_quality_check <- sleep_quality_check[,-which(names(sleep_quality_check) == "sleep_disorder_stress")]

#create the categorial sleep_disorder_stress variable
sleep_quality_check$sleep_disorder_stress <- cut(sleep_quality_check$Stress.Level, 
                                                  c(0, 3, 5, 9),
                                                  c("Low", "Moderate", "High"))

#shows the dimension of the data frame
dim(sleep_quality_check)
#displays the internal structure of the matrix, similar as the summary() function
str(sleep_quality_check)
#display the names of the variables
names(sleep_quality_check)

#checking the levels of the variables
levels(sleep_quality_check$sleep_disorder_stress)
levels(sleep_quality_check$Sleep.Disorder)

#2.1
#create a new data set including only the numerical variables
data_sep_num <- subset(sleep_quality_check, select = c(Age, Sleep.Duration, Stress.Level, 
                                                       Quality.of.Sleep, Physical.Activity.Level, Heart.Rate))
#describing the subset data frame created above
describe(data_sep_num)
summary(data_sep_num)

#shows the lowest value of the variable Age
#min(sleep_quality_check$Age)

#describing each numeric variable
summary(sleep_quality_check$Age)

summary(sleep_quality_check$Sleep.Duration)

summary(sleep_quality_check$Stress.Level)

summary(sleep_quality_check$Quality.of.Sleep)

summary(sleep_quality_check$Physical.Activity.Level)

summary(sleep_quality_check$Heart.Rate)



install.packages("psych")
library(psych)
describe(data_sep_num)

describeBy(sleep_quality_check$Sleep.Duration,
           group = sleep_quality_check$sleep_disorder_stress, digits = 4)

describeBy(data_sep_num, group = sleep_quality_check$Sleep.Disorder, digits = 4)

#shows the mean value of the variable Quality of Sleep based on the sleep_disorder_stress variable groups
tapply(sleep_quality_check$Quality.of.Sleep, sleep_quality_check$sleep_disorder_stress, mean)

#mean value of Quality of Sleep (the variable with values) based on the categories of variable Sleep.Disorder
aggregate(Quality.of.Sleep~Sleep.Disorder, sleep_quality_check, mean)


#2.2
#create the histograms for each numeric variable

hist(sleep_quality_check$Age, xlab = "Age of the people involved in the analysis")

hist(sleep_quality_check$Sleep.Duration, xlab = "Sleep Duration per night")

hist(sleep_quality_check$Stress.Level, xlab = "Stress Level")

hist(sleep_quality_check$Quality.of.Sleep, xlab = "Quality of Sleep")

hist(sleep_quality_check$Physical.Activity.Level, xlab = "Physical Activity Level")

hist(sleep_quality_check$Heart.Rate, xlab = "Heart Rate")


#the analysis of non numeric variables

plot(sleep_quality_check$Sleep.Disorder, xlab = "Sleep Disorder",
     col = c("lightblue", "green", "brown"))

plot(sleep_quality_check$sleep_disorder_stress, xlab = "The level of stress",
     col = c("green", "orange", "red"))

#identify the outliers using boxplots
boxplot(sleep_quality_check$Age, main = "boxplot Age",
        xlab = "Age of the people involved")
  
boxplot(sleep_quality_check$Sleep.Duration, main = "boxplot Sleep Duration",
        xlab = "Sleep Duration")

boxplot(sleep_quality_check$Stress.Level, main = "boxplot Stress Level",
        xlab = "Stress Level")

boxplot(sleep_quality_check$Quality.of.Sleep, main = "boxplot Quality of Sleep",
        xlab = "Quality of Sleep")

boxplot(sleep_quality_check$Physical.Activity.Level, main = "boxplot Physical Activity Level",
        xlab = "Physical Activity Level")

boxplot(sleep_quality_check$Heart.Rate, main = "boxplot Heart Rate",
        xlab = "Heart Rate")

#3
rm(tabelarea_datelor)

tabelarea_datelor <- table(sleep_quality_check$Sleep.Disorder,
                           sleep_quality_check$sleep_disorder_stress)
tabelarea_datelor


prop.table(tabelarea_datelor)

prop.table(tabelarea_datelor,1)

prop.table(tabelarea_datelor,2)

addmargins(prop.table(tabelarea_datelor))

#3.2

summary(tabelarea_datelor)

#3.3
#realized a Chi Square test
chisq.test(table(sleep_quality_check$sleep_disorder_stress))

chisq.test(table(sleep_quality_check$sleep_disorder_stress), 
           p = c(0.1, 0.3, 0.6))


chisq.test(table(sleep_quality_check$Sleep.Disorder))


chisq.test(table(sleep_quality_check$Sleep.Disorder), 
           p = c(0.1, 0.2, 0.7))

#4.1 
#covariance
cov(data_sep_num)

cor(data_sep_num)

cor(data_sep_num, method = 'spearman')

#covariance test
cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Age)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Sleep.Duration)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Quality.of.Sleep)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Physical.Activity.Level)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Heart.Rate)

#4.2
rm(regresie_lin_simpla)
rm(regresie_lin_multipla)

#the model for the simple linear regression

regresie_lin_simpla <- lm(Stress.Level~Quality.of.Sleep, sleep_quality_check)
regresie_lin_simpla
summary(regresie_lin_simpla)

regresie_lin_multipla <- lm(Stress.Level~Quality.of.Sleep+Age, sleep_quality_check)
regresie_lin_multipla
summary(regresie_lin_multipla)

#4.2.2

#the model for the non-linear regression

rm(regresie_neliniara)

regresie_neliniara <- lm(Stress.Level~Quality.of.Sleep+I(Quality.of.Sleep^2), sleep_quality_check)

regresie_neliniara

summary(regresie_neliniara)

#4.2.3
# Conducted ANOVA test between two regression models
anova(regresie_lin_simpla, regresie_lin_multipla)

#5.1
#mean test of the variable Stress Level vs a set value 0 or 5
t.test(sleep_quality_check$Stress.Level, mu = 0)
#5.2.1
t.test(sleep_quality_check$Stress.Level, mu = 5, alternative = "greater")

levels(sleep_quality_check$sleep_disorder_stress)

#5.2.2
#bartlett.test(Stress.Level~sleep_disorder_stress, sleep_quality_check,
              #Sleep.Disorder %in% c("Moderate", "High"))


bartlett.test(Stress.Level~Sleep.Disorder, sleep_quality_check,
              Sleep.Disorder %in% c("Insomnia", "Sleep Apnea"))

subset_sleep_quality_check <- sleep_quality_check[sleep_quality_check$Sleep.Disorder %in%
                                                    c("Insomnia", "Sleep Apnea"), ]

t.test(Stress.Level~Sleep.Disorder, subset_sleep_quality_check)

#5.2.3

objectaov <- aov(Stress.Level~sleep_disorder_stress, sleep_quality_check)
anova(objectaov)
coef(objectaov)

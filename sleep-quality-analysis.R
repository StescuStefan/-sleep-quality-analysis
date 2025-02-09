library(tidyverse)
library(dplyr)
library(tidyr)
library(stringr)
library(boot)
library(psych)
library(stats)

#conditii variabile
sleep <- read.csv("Sleep_health_and_lifestyle_dataset.csv")

sleep_quality_check <- subset(sleep, 
                              (Sleep.Duration >= 5.0 & Sleep.Duration <= 7.5) & 
                                (Stress.Level >= 2 & Stress.Level <= 9) & 
                                (Quality.of.Sleep >= 3 & Quality.of.Sleep <= 9), 
                              select = c(Person.ID, Gender, Age, Occupation, Sleep.Duration, Stress.Level, Quality.of.Sleep,
                                         Physical.Activity.Level, BMI.Category, Heart.Rate, Sleep.Disorder))

rm(sleep)
rm(sleep_quality_check)
rm(data_sep_num)
#colnames(sleep) <-make.names(colnames(sleep))
View(sleep_quality_check)


colSums(is.na(sleep_quality_check))

colnames(sleep)[which(colnames(sleep) == "")] <- "Person.ID"
colnames(sleep_quality_check)


write.csv(sleep_quality_check, "C:/Users/stesc/Documents/Proiect in R/sleep_quality_check.csv", row.names = FALSE)

sapply(sleep_quality_check, class)

sleep_quality_check$Sleep.Disorder <- as.factor(sleep_quality_check$Sleep.Disorder)

#sleep_quality_check$Gender <- as.character(sleep_quality_check$Gender)


#sleep_quality_check$overall_physical_state <- cut(sleep_quality_check$Heart.Rate, 
                                                  #c(0, 60, 100, 200),de
                                                  #c("Low", "Normal", "High"))

# Înlăturarea coloanei "overall_physical_state" din dataframe
#sleep_quality_check <- sleep_quality_check[,-which(names(sleep_quality_check) == "overall_physical_state")]


sleep_quality_check <- sleep_quality_check[,-which(names(sleep_quality_check) == "sleep_disorder_stress")]

sleep_quality_check$sleep_disorder_stress <- cut(sleep_quality_check$Stress.Level, 
                                                  c(0, 3, 5, 9),
                                                  c("Low", "Moderate", "High"))


dim(sleep_quality_check)

str(sleep_quality_check)

names(sleep_quality_check)

levels(sleep_quality_check$sleep_disorder_stress)
levels(sleep_quality_check$Sleep.Disorder)

#2.1

data_sep_num <- subset(sleep_quality_check, select = c(Age, Sleep.Duration, Stress.Level, 
                                                       Quality.of.Sleep, Physical.Activity.Level, Heart.Rate))

describe(data_sep_num)


summary(data_sep_num)

#min(sleep_quality_check$Age)

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

tapply(sleep_quality_check$Quality.of.Sleep, sleep_quality_check$sleep_disorder_stress, mean)

aggregate(Quality.of.Sleep~Sleep.Disorder, sleep_quality_check, mean)


#2.2

hist(sleep_quality_check$Age, xlab = "Age of the people involved in the analysis")

hist(sleep_quality_check$Sleep.Duration, xlab = "Sleep Duration per night")

hist(sleep_quality_check$Stress.Level, xlab = "Stress Level")

hist(sleep_quality_check$Quality.of.Sleep, xlab = "Quality of Sleep")

hist(sleep_quality_check$Physical.Activity.Level, xlab = "Physical Activity Level")

hist(sleep_quality_check$Heart.Rate, xlab = "Heart Rate")


#analiza grafica nenumerice

plot(sleep_quality_check$Sleep.Disorder, xlab = "Tulburările de somn întâlnite",
     col = c("lightblue", "green", "brown"))

plot(sleep_quality_check$sleep_disorder_stress, xlab = "Nivelul de stres în
     funcție de probleme preexistente",
     col = c("green", "orange", "red"))

#identificare outlieri

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
chisq.test(table(sleep_quality_check$sleep_disorder_stress))

chisq.test(table(sleep_quality_check$sleep_disorder_stress), 
           p = c(0.1, 0.3, 0.6))


chisq.test(table(sleep_quality_check$Sleep.Disorder))


chisq.test(table(sleep_quality_check$Sleep.Disorder), 
           p = c(0.1, 0.2, 0.7))

#4.1 

cov(data_sep_num)

cor(data_sep_num)

cor(data_sep_num, method = 'spearman')


cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Age)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Sleep.Duration)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Quality.of.Sleep)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Physical.Activity.Level)

cor.test(sleep_quality_check$Stress.Level, sleep_quality_check$Heart.Rate)

#4.2
rm(regresie_lin_simpla)
rm(regresie_lin_multipla)



regresie_lin_simpla <- lm(Stress.Level~Quality.of.Sleep, sleep_quality_check)
regresie_lin_simpla
summary(regresie_lin_simpla)

regresie_lin_multipla <- lm(Stress.Level~Quality.of.Sleep+Age, sleep_quality_check)
regresie_lin_multipla
summary(regresie_lin_multipla)


#4.2.2

rm(regresie_neliniara)

regresie_neliniara <- lm(Stress.Level~Quality.of.Sleep+I(Quality.of.Sleep^2), sleep_quality_check)


# Crearea unui plot cu datele reale și valorile prezise
#plot(sleep_quality_check$Quality.of.Sleep, sleep_quality_check$Stress.Level,
    # main = "Regresie Neliniară: Nivelul de Stres vs. Calitatea Somnului",
    # xlab = "Calitatea Somnului",
     #ylab = "Nivelul de Stres",
    # pch = 19, col = "blue")  # Punctele pentru datele reale

# Adăugarea liniei de regresie
#lines(sleep_quality_check$Quality.of.Sleep, predict(regresie_neliniara), col = "red", lwd = 2)


#regresie_neliniara <- lm(Quality.of.Sleep~Stress.Level+I(Stress.Level^2), sleep_quality_check)
regresie_neliniara

summary(regresie_neliniara)

#4.2.3

anova(regresie_lin_simpla, regresie_lin_multipla)



#5.1
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


#table(sleep_quality_check$Sleep.Disorder, sleep_quality_check$Stress.Level)

# Test chi-pătrat
#chisq.test(sleep_quality_check$Sleep.Disorder, sleep_quality_check$Stress.Level)

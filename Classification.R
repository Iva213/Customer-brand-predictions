## Classification - predict which brand of products customers prefer

## Set working directory 
setwd("/Users/Iva/Desktop/Data Analytics & Machine Learning/R/Data Analytics II")

library(caret)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(mltools)

set.seed(1234)
################################################

## Read in files 
complete <- read.csv("CompleteResponses.csv")
incomplete <- read.csv("SurveyIncomplete.csv")

brand.1 <- complete %>%
  filter(brand == 1) 
hist(brand.1$car)

brand.0 <- complete %>%
  filter(brand == 0)
hist(brand.0$car)

brand.0 %>%
  group_by(car) %>%
  #group_by(elevel) %>%
  count()

## Initial visualisation -- look for weird shit 

ggplot(complete, aes(x = as.factor(brand), y = elevel)) +
  geom_boxplot()

#Histogram of all variables 
par(mfrow=c(3,3),oma=c(0,0,3,0))                         
histogram <- lapply(names(brand.0), function(x) hist(complete[,x],
                                        xlab = x, 
                                        border="blue", 
                                        col="green",
                                        main = c("Histogram of", x)))

#Change between geom_bar and geom_histogram for categorical VS numerical (independent) variables 
ggplot(complete, aes(as.factor(car), fill = as.factor(brand), color = as.factor(brand))) +
  geom_bar(position = position_dodge(width=0.05)) +
  scale_fill_discrete(name="Brand", labels = c("Acer", "Sony")) +
  scale_color_discrete(name="Brand", labels = c("Acer", "Sony"))

ggplot(complete, aes(age, salary, col = brand)) +
  geom_jitter(size = 4) +
  labs(col = "Brand")

ggplot(complete, aes(brand, age)) +
  geom_point()

ggplot(brand.1, aes(car)) +
  geom_bar() + ylim(0, 400)

################################################

## Visualise correlation matrix 

cormat <- cor(complete); round(cormat, 2)

# Get lower triangle of the correlation matrix
get.lower.tri <- function(cormat) {
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

lower.tri <- get.lower.tri(cormat)
lower.tri

melt.cormat <- melt(lower.tri, na.rm = TRUE)

ggplot(melt.cormat, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(col = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed()

################################################

## Switch numerical variables to categorical

complete$elevel <- as.factor(complete$elevel)
complete$car <- as.factor(complete$car)
complete$zipcode <- as.factor(complete$zipcode)
complete$brand <- as.factor(complete$brand)

################################################

## C5.0 model 

#Split data - 75-25
in.train <- createDataPartition(complete$brand, p = 0.75, list = FALSE)
training <- complete[in.train, ]
testing <- complete[-in.train, ]

#Train model
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, verboseIter = TRUE)

c50.mod <- train(brand ~ ., data = training, method = "C5.0",
                 trControl = ctrl,
                 tuneLength = 2,
                 preProc = c("center", "scale"))

pred <- predict(c50.mod, testing)

testing.pred <- cbind(testing, pred)

plot(varImp(c50.mod))

#Confusion matrix - data = predicted; reference = actual 
conmat <- confusionMatrix(data = testing.pred$pred, reference = testing.pred$brand)

################################################

## Bin salary and age 

complete$salary <- bin_data(complete$salary, bins = 13, binType = "explicit", 
                            boundaryType = "lcro]" ,
                            returnDT = FALSE)

complete$age <- bin_data(complete$age, bins = 6, binType = "explicit", 
                         boundaryType = "lcro]" ,
                         returnDT = FALSE)

ggplot(complete, aes(age, salary, col = brand)) +
  geom_jitter()

################################################

## Run C5.0 again with BINNED salary and age 

in.train <- createDataPartition(complete$brand, p = 0.75, list = FALSE)
training <- complete[in.train, ]
testing <- complete[-in.train, ]

c50.mod.bin <- train(brand ~ ., data = training, 
                 method = "C5.0",
                 trControl = ctrl,
                 tuneLength = 2,
                 preProc = c("center", "scale"))

plot(varImp(c50.mod.bin))

pred.bin <- predict(c50.mod.bin, testing)

testing.pred.bin <- cbind(testing, pred.bin)

ggplot(testing.pred.bin, aes(age, salary, col = interaction(brand, pred.bin, sep="-", 
                                                            lex.order = TRUE))) +
  geom_jitter(size = 5) +
  labs(col = "Brand-prediction")

#Confusion matrix - data = predicted; reference = actual 
conmat.bin <- confusionMatrix(data = testing.pred.bin$pred, reference = testing.pred.bin$brand)
plot(conmat.bin$table)

#C5.0 - ONLY USING BINNED SALARY & AGE

training.sal.age <- training[c(1:2, 7)]
testing.sal.age <- testing[c(1:2, 7)]

c50.mod.2var <- train(brand ~ ., data = training.sal.age, 
                     method = "C5.0",
                     trControl = ctrl,
                     tuneLength = 2,
                     preProc = c("center", "scale"))

plot(varImp(c50.mod.2var))

pred.2var <- predict(c50.mod.2var, testing.sal.age)

testing.pred.2var <- cbind(testing.sal.age, pred.2var)

#Confusion matrix - data = predicted; reference = actual 
conmat.2var <- confusionMatrix(data = testing.pred.2var$pred, reference = testing.pred.2var$brand)

postResample(testing.pred.2var$pred, testing.pred.2var$brand)

ggplot(testing.pred.2var, aes(age, salary, col = interaction(brand, pred.2var, sep="-", 
                                                            lex.order = TRUE))) +
  geom_jitter(size = 5) +
  labs(col = "Brand-prediction")

################################################

## Random forest 

rf.mod.2var <- train(brand ~ ., data = training.sal.age, 
                     method = "rf",
                     trControl = ctrl,
                     preProc = c("center", "scale"))

plot(varImp(rf.mod.2var))

pred.rf.2var <- predict(rf.mod.2var, testing.sal.age)

testing.pred.rf.2var <- cbind(testing.sal.age, pred.rf.2var)

ggplot(testing.pred.rf.2var, aes(age, salary, col = interaction(brand, pred.rf.2var, sep="-", 
                                                             lex.order = TRUE))) +
  geom_jitter(size = 5) +
  labs(col = "Brand-prediction")

conmat.rf.2var <- confusionMatrix(data = testing.pred.rf.2var$pred, reference = testing.pred.rf.2var$brand)

################################################

## Make prediction on INCOMPLETE dataset

#First, repeat feature selection you did on complete dataset

#Switch numerical variables to categorical
incomplete$elevel <- as.factor(incomplete$elevel)
incomplete$car <- as.factor(incomplete$car)
incomplete$zipcode <- as.factor(incomplete$zipcode)

#Bin salary and age 
incomplete$salary <- bin_data(incomplete$salary, bins = 13, binType = "explicit", 
                            boundaryType = "lcro]" ,
                            returnDT = FALSE)

incomplete$age <- bin_data(incomplete$age, bins = 6, binType = "explicit", 
                         boundaryType = "lcro]" ,
                         returnDT = FALSE)

incomplete.pred <- predict(c50.mod.2var, incomplete)

incomplete.updated <- cbind(incomplete[-7], incomplete.pred)
colnames(incomplete.updated)[7] <- "brand" 

ggplot(incomplete.updated, aes(age, salary, col = incomplete.pred)) +
  geom_jitter(size = 2) +
  labs(col = "Brand")


t.test(as.numeric(testing.pred.2var$pred.2var), as.numeric(incomplete.updated$brand))
################################################

##Plot of all 15,000 responses 

training.sal.age <- training[c(1:2, 7)]
testing.sal.age <- testing[c(1:2, 7)]

combined <- rbind(training.sal.age, testing.sal.age)
combined <- rbind(combined, incomplete.updated[c(1:2, 7)])

ggplot(combined, aes(age, salary, col = brand)) +
  geom_jitter(size = 2) +
  labs(col = "Brand")


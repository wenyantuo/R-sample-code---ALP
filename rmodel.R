library(caret)
library(rlang)
library(psych)
library(tidyverse)#
library(tidyr)#
library(dplyr)#
library(jsonlite)
library(DataExplorer)#
library(ggthemes)#
library(cowplot)
library(dummies)
library(h2o)
library(lubridate)
library(magrittr)
library(lattice)
library(flexmix)
library(Matrix)


#read the original csv dataset
train.cust <- read.csv("Customer4ethan.csv")
train.cust$purchase <- ifelse(train.cust$purchase>0,1,0)

#split the train dataset into train and test
train.index <- sample(1:nrow(train.cust),0.8*nrow(train.cust),replace = FALSE)
train.train <- train.cust[train.index,]
train.test <- train.cust[-train.index,]

#oversampling
train.rose11 <- ovun.sample(purchase~., data=train.train,method="both" ,seed=123)$data
train.rose0.2 <- ovun.sample(purchase~., data=train.train, seed=123,p=0.2)$data
train.rose0.1 <- ovun.sample(purchase~., data=train.train, seed=123,p=0.1)$data
train.rose0.15 <- ovun.sample(purchase~., data=train.train, seed=123,p=0.15)$data
train.rose0.4 <- ovun.sample(purchase~., data=train.train, seed=123,p=0.4)$data
table(train.rose11$purchase)
table(train.rose0.2$purchase)
table(train.rose0.1$purchase)
table(train.rose0.15$purchase)
table(train.rose0.4$purchase)

#write these datasets
#write.csv(train.rose11,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose11.csv")
#write.csv(train.rose0.1,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.1.csv")
#write.csv(train.rose0.2,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.2.csv")
#write.csv(train.rose0.15,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.15.csv")
#write.csv(train.rose0.4,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.4.csv")
#write.csv(train.test,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.test0.2.csv")
#write.csv(train.cust,"C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.cust.csv")

#load h2o
h2o.init(nthreads = -1,max_mem_size = "8G")
train.train <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.train.csv")
train.rose11 <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose11.csv")
train.rose0.1 <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.1.csv")
train.rose0.15 <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.15.csv")
train.rose0.2 <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.2.csv")
train.rose0.4 <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.rose0.4.csv")
data.test <- h2o.importFile("C:/Users/Chen Yushuang/Desktop/UMD/Action Learning/train.test0.2.csv")

data.lm.train <- train.train[train.train$transactionRevenue>0,]
data.lm.test <- data.test[data.test$transactionRevenue>0,]

v1 <- colnames(train.rose0.1)
features.logit <- v1[-c(1,2,5,6)]
y <- "purchase"
#modeling
logit_model.no <- h2o.glm(x=features.logit,y=y,training_frame = train.train,nfolds = 5,
                         model_id = "logistic.no",family = "binomial",link="logit",alpha = 0.5)
logit_model.0 <- h2o.glm(x=features.logit,y=y,training_frame = train.rose11,nfolds = 5,
                         model_id = "logistic.0",family = "binomial",link="logit")
logit_model.0.1 <- h2o.glm(x=features.logit,y=y,training_frame = train.rose0.1,nfolds = 5,
                           model_id = "logistic.0.1",family = "binomial",link="logit",lambda_search = TRUE)
logit_model.0.15 <- h2o.glm(x=features.logit,y=y,training_frame = train.rose0.15,nfolds = 5,
                            model_id = "logistic.0.15",family = "binomial",link="logit",lambda_search = TRUE)
logit_model.0.15.F <- h2o.glm(x=features.logit,y=y,training_frame = train.rose0.15,
                              model_id = "logistic.0.15.F",family = "binomial",link="logit")
logit_model.0.2 <- h2o.glm(x=features.logit,y=y,training_frame = train.rose0.2,nfolds = 5,
                           model_id = "logistic.0.2",family = "binomial",lambda_search = TRUE)
logit_model.0.4 <- h2o.glm(x=features.logit,y=y,training_frame = train.rose0.4,nfolds = 5,
                           model_id = "logistic.0.4",family = "binomial",lambda_search = TRUE)
#the performance of models
glm_perfno <- h2o.performance(model = logit_model.no,
                             newdata = data.test)
glm_perf0 <- h2o.performance(model = logit_model.0,
                             newdata = data.test)
glm_perf0.1 <- h2o.performance(model = logit_model.0.1,
                               newdata = data.test)
glm_perf0.15 <- h2o.performance(model = logit_model.0.15,
                                newdata = data.test)
glm_perf0.15.F <- h2o.performance(model = logit_model.0.15.F,
                                  newdata = data.test)
glm_perf0.2 <- h2o.performance(model = logit_model.0.2,
                               newdata = data.test)
glm_perf0.4 <- h2o.performance(model = logit_model.0.4,
                               newdata = data.test)

#metric to measure models
h2o.confusionMatrix(glm_perfno)
h2o.confusionMatrix(glm_perf0)
h2o.confusionMatrix(glm_perf0.1)
h2o.confusionMatrix(glm_perf0.15)
h2o.confusionMatrix(glm_perf0.15.F)
h2o.confusionMatrix(glm_perf0.2)
h2o.confusionMatrix(glm_perf0.4)
glm.pred1 <- h2o.predict(logit_model.0.2,newdata = data.test)[3]
glm.predno <- h2o.predict(logit_model.no,newdata = data.test)[3]

#other classification models

#linear
#data.lm.train <- train.train[train.train$transactionRevenue>0,]
#data.lm.test <- data.test[data.test$transactionRevenue>0,]
test.lm <- data.test
test.lm$logR <- log(test.lm$transactionRevenue+1)
data.lm.train$logR <- log(data.lm.train$transactionRevenue+1)
v.lm.test <- colnames(test.lm)
test.lm <- test.lm[,-c(1,2,5,6)]
v.lm <- colnames(data.lm.train)
v.lm <- v.lm[-c(1,2,5,6,44)]
y2 <- "logR"
lm.1 <- h2o.glm(y=y2,x=v.lm,training_frame = data.lm.train,
                model_id = "lm.1",family = "gaussian",link = "identity",lambda = 0,alpha = 1) 
summ.lm1 <- summary(lm.1)
lm.pred1 <- h2o.predict(lm.1,newdata = data.test)
pred1 <- lm.pred1*glm.pred1
pred2 <- lm.pred1*glm.predno
RMSE1 <- RMSE(pred1,test.lm$logR)
RMSE2 <- RMSE(pred2,test.lm$logR)

###








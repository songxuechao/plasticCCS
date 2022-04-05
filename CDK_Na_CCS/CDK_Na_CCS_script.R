library(tidyverse)
library(caret)
library(usdm)
library(e1071)
library(xgboost)

#remove descriptors with near zero variance
Na <- read.csv("Na.csv", header = T, sep = ",")
Na <- Na[,-nearZeroVar(Na)]

#calculate R2 between CCS and 207 descriptors
regre_coeff <- data.frame(variable = colnames(Na)[3:209], r2 = NA)
for(i in 1:nrow(regre_coeff)) {
  regre_coeff$r2[i] <- R2(Na$CCS, Na[,i+2])
}

#select descriptors with R2 > 0.36
regre_coeff_0.6 <- regre_coeff %>% filter(r2 > 0.36)
Na_65 <- Na[,c("NAME","CCS",regre_coeff_0.6$variable)]


#scale x variables and log CCS
scaled_Na <- as_tibble(scale(Na[,3:209]))
scaled_Na <- cbind(Na$NAME, log(Na$CCS), scaled_Na)
colnames(scaled_Na)[1:2] <-c("name", "ccs")
rownames(scaled_Na) <- scaled_Na$name
scaled_Na <- scaled_Na[,-1]


#divide into train and test data set
index <- sample(2, nrow(scaled_Na), replace = T, prob = c(0.7, 0.3))
trainNa_207 <- scaled_Na[index==1,]
testNa_207 <- scaled_Na[index==2,]

scaled_Na_65 <- scaled_Na[,c("ccs", regre_coeff_0.6$variable)]
trainNa_65 <- scaled_Na_65[rownames(trainNa_207),]
testNa_65 <- scaled_Na_65[rownames(testNa_207),]
         

#build SVM model with 207 descriptors
tune.svm(ccs ~., data = trainNa_207, gamma = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/207, cost = 2^(0:8))
svm_207 <- svm(ccs~.,data = trainNa_207, gamma = 0.0004830918, cost = 16)

# predict CCS of testing set
svm_pred_207 <- predict(svm_207, testNa_207[,-1])
rmsep <- RMSE(exp(svm_pred_207),exp(testNa_207$ccs))
r2_test <- R2(exp(svm_pred_207),exp(testNa_207$ccs))

pred_Na_svm_207 <- data.frame(pred = exp(svm_pred_207), empirical = exp(testNa_207$ccs))
pred_Na_svm_207 <- pred_Na_svm_207 %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_Na_svm_207$error<2 & pred_Na_svm_207$error>-2)/length(pred_Na_svm_207$error)
sum(pred_Na_svm_207$error<3 & pred_Na_svm_207$error>-3)/length(pred_Na_svm_207$error)
sum(pred_Na_svm_207$error<5 & pred_Na_svm_207$error>-5)/length(pred_Na_svm_207$error)
median(abs(pred_Na_svm_207$error))

#export the prediction results
pred_Na_svm_207 <- pred_Na_svm_207 %>% mutate(data="CDK_207", algorithm = "SVM")
write.csv(pred_Na_svm_207, file = "pred_Na_svm_207.csv")



#try XGBoost with 207 descriptors

#optimize the xgboost parameters
hyper_grid <- expand.grid(eta = c(0.01, 0.05, 0.1, 0.3),
                          max_depth = c(3,5,7),
                          min_child_weight = c(1,3,5),
                          subsample = c(0.6, 0.7, 0.8, 0.9),
                          colsample_bytree = c(0.6, 0.7, 0.8, 0.9),
                          nrounds = 0, 
                          RMSE = 0)

for(i in 1:nrow(hyper_grid)) {
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  set.seed(206)
  xgb_tune <- xgb.cv(
    params = params,
    data = as.matrix(trainNa_207[,-1]),
    label = trainNa_207[,1],
    nrounds = 500,
    nfold = 10,
    objective = "reg:squarederror",
    verbose = 0,
    early_stopping_rounds = 10
  )
  hyper_grid$nrounds[i] <- which.min(xgb_tune$evaluation_log$test_rmse_mean)
  hyper_grid$RMSE[i] <- min(xgb_tune$evaluation_log$test_rmse_mean)
}


#build xgboost model with 207 descriptors
hyper_grid %>% arrange(RMSE) %>% head(10)

set.seed(207)
xgb_207 <- xgboost(data = as.matrix(trainNa_207[,-1]), label = trainNa_207[,1],
                   eta = 0.05, max_depth = 5, 
                   min_child_weight = 3, 
                   subsample = 0.7, colsample_bytree = 0.9,
                   nrounds = 395,
                   objective = "reg:squarederror")

#predict the CCS of testing set using xgb_207
pred_Na_xgb_207 <- predict(xgb_207, as.matrix(testNa_207[,-1]))
pred_Na_xgb_207 <- data.frame(pred = exp(pred_Na_xgb_207), empirical = exp(testNa_207$ccs))
pred_Na_xgb_207 <- pred_Na_xgb_207 %>% mutate(error = (pred - empirical)/empirical*100)


rmsep <- RMSE(pred_Na_xgb_207$pred, pred_Na_xgb_207$empirical)
r2_test <-R2(pred_Na_xgb_207$pred, pred_Na_xgb_207$empirical)
sum(pred_Na_xgb_207$error<2 & pred_Na_xgb_207$error>-2)/length(pred_Na_xgb_2076$error)
sum(pred_Na_xgb_207$error<3 & pred_Na_xgb_207$error>-3)/length(pred_Na_xgb_207$error)
sum(pred_Na_xgb_207$error<5 & pred_Na_xgb_207$error>-5)/length(pred_Na_xgb_207$error)
median(abs(pred_Na_xgb_207$error))

pred_Na_xgb_207 <- pred_Na_xgb_207 %>% mutate(data="CDK_207", algorithm = "XGBoost")
write.csv(pred_Na_xgb_207, file = "pred_Na_xgb_207.csv")



#build SVM model with 65 descriptors
tune.svm(ccs ~., data = trainNa_65, gamma = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/65, cost = 2^(0:8))
svm_65 <- svm(ccs~.,data = trainNa_65, gamma = 0.001538462, cost = 64)


svm_pred_65 <- predict(svm_65, testNa_65[,-1])
rmsep <- RMSE(exp(svm_pred_65),exp(testNa_65$ccs))
r2_test <- R2(exp(svm_pred_65),exp(testNa_65$ccs))

pred_Na_svm_65 <- data.frame(pred = exp(svm_pred_65), empirical = exp(testNa_65$ccs))
pred_Na_svm_65 <- pred_Na_svm_65 %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_Na_svm_65$error<2 & pred_Na_svm_65$error>-2)/length(pred_Na_svm_65$error)
sum(pred_Na_svm_65$error<3 & pred_Na_svm_65$error>-3)/length(pred_Na_svm_65$error)
sum(pred_Na_svm_65$error<5 & pred_Na_svm_65$error>-5)/length(pred_Na_svm_65$error)
median(abs(pred_Na_svm_65$error))



#try XGBoost with 65 descriptors
#optimize the xgboost parameters
for(i in 1:nrow(hyper_grid)) {
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  set.seed(64)
  xgb_tune <- xgb.cv(
    params = params,
    data = as.matrix(trainNa_65[,-1]),
    label = trainNa_65[,1],
    nrounds = 500,
    nfold = 10,
    objective = "reg:squarederror",
    verbose = 0,
    early_stopping_rounds = 10
  )
  hyper_grid$nrounds[i] <- which.min(xgb_tune$evaluation_log$test_rmse_mean)
  hyper_grid$RMSE[i] <- min(xgb_tune$evaluation_log$test_rmse_mean)
}


#build xgboost model with 65 descriptors
hyper_grid %>% arrange(RMSE) %>% head(10)

set.seed(65)
xgb_65 <- xgboost(data = as.matrix(trainNa_65[,-1]), label = trainNa_65[,1],
                  eta = 0.05, max_depth = 7, 
                  min_child_weight = 1, 
                  subsample = 0.7, colsample_bytree = 0.9,
                  nrounds = 231,
                  objective = "reg:squarederror")

#predict CCS of testing set using xgb_65
pred_Na_xgb_65 <- predict(xgb_65, as.matrix(testNa_65[,-1]))
pred_Na_xgb_65 <- data.frame(pred = exp(pred_Na_xgb_65), empirical = exp(testNa_65$ccs))
pred_Na_xgb_65 <- pred_Na_xgb_65 %>% mutate(error = (pred - empirical)/empirical*100)


rmsep <- RMSE(pred_Na_xgb_65$pred, pred_Na_xgb_65$empirical)
r2_test <-R2(pred_Na_xgb_65$pred, pred_Na_xgb_65$empirical)
sum(pred_Na_xgb_65$error<2 & pred_Na_xgb_65$error>-2)/length(pred_Na_xgb_65$error)
sum(pred_Na_xgb_65$error<3 & pred_Na_xgb_65$error>-3)/length(pred_Na_xgb_65$error)
sum(pred_Na_xgb_65$error<5 & pred_Na_xgb_65$error>-5)/length(pred_Na_xgb_65$error)
median(abs(pred_Na_xgb_65$error))

pred_Na_xgb_65 <- pred_Na_xgb_65 %>% mutate(data="CDK_65", algorithm = "XGBoost")
write.csv(pred_Na_xgb_65, file = "pred_H_xgb_65.csv")




#build SVM model only with TWCCS
trainNa_207_twccs <- read.csv("trainNa_207_twccs.csv", sep = ",", header = T)
rownames(trainNa_207_twccs) <- trainNa_207_twccs$name
trainNa_207_twccs <- trainNa_207_twccs[,-1]

tune.svm(ccs ~., data = trainNa_207_twccs, gamma = c(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)/207, cost = 2^(-8:8))
svm_207_twccs <- svm(ccs~.,data = trainNa_207_twccs, gamma = 0.00024154595, cost = 64)

# predict CCS of testing set
svm_pred_twccs <- predict(svm_207_twccs, testNa_207[,-1])
plot(exp(svm_pred_twccs),exp(testNa_207$ccs))
rmsep <- RMSE(exp(svm_pred_twccs),exp(testNa_207$ccs))
r2_test <- R2(exp(svm_pred_twccs),exp(testNa_207$ccs))

pred_Na_svm_twccs <- data.frame(pred = exp(svm_pred_twccs), empirical = exp(testNa_207$ccs))
pred_Na_svm_twccs <- pred_Na_svm_twccs %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_Na_svm_twccs$error<2 & pred_Na_svm_twccs$error>-2)/length(pred_Na_svm_twccs$error)
sum(pred_Na_svm_twccs$error<3 & pred_Na_svm_twccs$error>-3)/length(pred_Na_svm_twccs$error)
sum(pred_Na_svm_twccs$error<5 & pred_Na_svm_twccs$error>-5)/length(pred_Na_svm_twccs$error)
median(abs(pred_Na_svm_twccs$error))








#build SVM with 15 descriptors from AllCCS
desc_allccs <- c("MW", "VAdjMat", "FMF", "nAtomLAC", "khs.ssCH2", 
                 "Kier3", "nAtom", "VP.0", "C2SP3", "VP.1", 
                 "MDEC.12", "ALogp2", "VP.2","nAtomLC","AMR")

scaled_Na_allccs <- scaled_Na[,c("ccs", desc_allccs)]
trainNa_allccs <- scaled_Na_allccs[rownames(trainNa_207),]
testNa_allccs <- scaled_Na_allccs[rownames(testNa_207),]

tune.svm(ccs ~., data = trainNa_allccs, gamma = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/15, cost = 2^(0:8))
svm_allccs <- svm(ccs~.,data = trainNa_allccs, gamma = 0.003333333, cost = 128)

#predict CCS of testing set
svm_pred_allccs <- predict(svm_allccs, testNa_allccs[,-1])
plot(exp(svm_pred_allccs),exp(testNa_allccs$ccs))
rmsep <- RMSE(exp(svm_pred_allccs),exp(testNa_allccs$ccs))
r2_test <- R2(exp(svm_pred_allccs),exp(testNa_allccs$ccs))

pred_Na_allccs <- data.frame(pred = exp(svm_pred_allccs), empirical = exp(testNa_allccs$ccs))
pred_Na_allccs <- pred_Na_allccs %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_Na_allccs$error<2 & pred_Na_allccs$error>-2)/length(pred_Na_allccs$error)
sum(pred_Na_allccs$error<3 & pred_Na_allccs$error>-3)/length(pred_Na_allccs$error)
sum(pred_Na_allccs$error<5 & pred_Na_allccs$error>-5)/length(pred_Na_allccs$error)
median(abs(pred_Na_allccs$error))






# predict the CCS of unknowns, an example is shown below.
#first get the mean and sd of each descriptors
variable <- colnames(trainNa_207)[-1]
Na_207des <- Na[,variable]
mean <- apply(Na_207des, 2, mean)
sd <- apply(Na_207des, 2, sd)

#then import and scale the descriptors of example by mean and sd obtained above
example <- read.csv("example.csv", header = T, sep = ",", row.names = "name")
example <- example[,variable]

example_center <- sweep(example, 2, mean, "-")
example_scaled <- sweep(example_center, 2, sd, "/")

#predict the CCS of example
pred_example <- predict(svm_207, example_scaled)
pred_example <- data.frame(pred = exp(pred_example))

write.csv(pred_example, file = "pred_example.csv")

























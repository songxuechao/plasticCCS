library(tidyverse)
library(caret)
library(e1071)
library(xgboost)
library(usdm)

#remove descriptors with near zero variance
H <- read.csv("H.csv", header = T, sep = ",")
H <- H[,-nearZeroVar(H)]

#calculate regression coefficient between CCS and 206 descriptors
regre_coeff <- data.frame(variable = colnames(H)[3:208], r2 = NA)
for(i in 1:nrow(regre_coeff)) {
  regre_coeff$r2[i] <- R2(H$CCS, H[,i+2])
}

#select descriptors with r > 0.6
regre_coeff_0.6 <- regre_coeff %>% filter(r2 > 0.36)
H_84 <- H[,c("name","CCS",regre_coeff_0.6$variable)]
colnames(H_84)[2] <- "ccs"


#scale x variables and log CCS
scaled_H <- as_tibble(scale(H[,3:208]))
scaled_H <- cbind(H$name, log(H$CCS), scaled_H)
colnames(scaled_H)[1:2] <-c("name", "ccs")
rownames(scaled_H) <- scaled_H$name
scaled_H <- scaled_H[,-1]


#divide into train and test data set
index <- sample(2, nrow(scaled_H), replace = T, prob = c(0.7, 0.3))
trainH_206 <- scaled_H[index==1,]
testH_206 <- scaled_H[index==2,]

scaled_H_84 <- scaled_H[,c("ccs", regre_coeff_0.6$variable)]
trainH_84 <- scaled_H_84[rownames(trainH_206),]
testH_84 <- scaled_H_84[rownames(testH_206),]





#build SVM model with 206 descriptors
tune.svm(ccs ~., data = trainH_206, gamma = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/206, cost = 2^(0:8))
svm_206 <- svm(ccs~.,data=trainH_206, gamma = 0.00002427184, cost = 128)


# predict CCS of testing set
svm_pred_206 <- predict(svm_206, testH_206[,-1])
plot(exp(svm_pred_206),exp(testH_206$ccs))
rmsep <- RMSE(exp(svm_pred_206),exp(testH_206$ccs))
r2_test <- R2(exp(svm_pred_206),exp(testH_206$ccs))

pred_H_svm_206 <- data.frame(pred = exp(svm_pred_206), empirical = exp(testH_206$ccs))
pred_H_svm_206 <- pred_H_svm_206 %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_H_svm_206$error<2 & pred_H_svm_206$error>-2)/length(pred_H_svm_206$error)
sum(pred_H_svm_206$error<3 & pred_H_svm_206$error>-3)/length(pred_H_svm_206$error)
sum(pred_H_svm_206$error<5 & pred_H_svm_206$error>-5)/length(pred_H_svm_206$error)
median(abs(pred_H_svm_206$error))

#export the prediction results
pred_H_svm_206 <- pred_H_svm_206 %>% mutate(data="CDK_206", algorithm = "SVM")
write.csv(pred_H_svm_206, file = "pred_H_svm_206.csv")




#try XGBoost with 206 descriptors

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
  
  set.seed(205)
  xgb_tune <- xgb.cv(
    params = params,
    data = as.matrix(trainH_206[,-1]),
    label = trainH_206[,1],
    nrounds = 500,
    nfold = 10,
    objective = "reg:squarederror",
    verbose = 0,
    early_stopping_rounds = 10
  )
  hyper_grid$nrounds[i] <- which.min(xgb_tune$evaluation_log$test_rmse_mean)
  hyper_grid$RMSE[i] <- min(xgb_tune$evaluation_log$test_rmse_mean)
}



#build xgboost model with 206 descriptors
hyper_grid %>% arrange(RMSE) %>% head(10)

set.seed(206)
xgb_206 <- xgboost(data = as.matrix(trainH_206[,-1]), label = trainH_206[,1],
                   eta = 0.05, max_depth = 5, 
                   min_child_weight = 3, 
                   subsample = 0.8, colsample_bytree = 0.7,
                   nrounds = 335,
                   objective = "reg:squarederror")

#predict CCS of testing set using xgb_206
pred_H_xgb_206 <- predict(xgb_206, as.matrix(testH_206[,-1]))
pred_H_xgb_206 <- data.frame(pred = exp(pred_H_xgb_206), empirical = exp(testH_206$ccs))
pred_H_xgb_206 <- pred_H_xgb_206 %>% mutate(error = (pred - empirical)/empirical*100)


rmsep <- RMSE(pred_H_xgb_206$pred, pred_H_xgb_206$empirical)
r2_test <-R2(pred_H_xgb_206$pred, pred_H_xgb_206$empirical)
sum(pred_H_xgb_206$error<2 & pred_H_xgb_206$error>-2)/length(pred_H_xgb_206$error)
sum(pred_H_xgb_206$error<3 & pred_H_xgb_206$error>-3)/length(pred_H_xgb_206$error)
sum(pred_H_xgb_206$error<5 & pred_H_xgb_206$error>-5)/length(pred_H_xgb_206$error)
median(abs(pred_H_xgb_206$error))

pred_H_xgb_206 <- pred_H_xgb_206 %>% mutate(data="CDK_206", algorithm = "XGBoost")
write.csv(pred_H_xgb_206, file = "pred_H_xgb_206.csv")



#build SVM model with 84 descriptors...............................................................
tune.svm(ccs ~., data = trainH_84, gamma = c(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/84, cost = 2^(0:8))
svm_84 <- svm(ccs~.,data=trainH_84, gamma = 0.001190476, cost = 32)

# predict CCS of testing set
svm_pred_84 <- predict(svm_84, testH_84[,-1])
rmsep <- RMSE(exp(svm_pred_84),exp(testH_84$ccs))
r2_test <- R2(exp(svm_pred_84),exp(testH_84$ccs))

pred_H_svm_84 <- data.frame(pred = exp(svm_pred_84), empirical = exp(testH_84$ccs))
pred_H_svm_84 <- pred_H_svm_84 %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_H_svm_84$error<2 & pred_H_svm_84$error>-2)/length(pred_H_svm_84$error)
sum(pred_H_svm_84$error<3 & pred_H_svm_84$error>-3)/length(pred_H_svm_84$error)
sum(pred_H_svm_84$error<5 & pred_H_svm_84$error>-5)/length(pred_H_svm_84$error)
median(abs(pred_H_svm_84$error))

pred_H_svm_84 <- pred_H_svm_206 %>% mutate(data="CDK_84", algorithm = "SVM")
write.csv(pred_H_svm_84, file = "pred_H_SVM_84.csv")


#build XGBoost with 84 descriptors
#optimize the parameters
for(i in 1:nrow(hyper_grid)) {
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  
  set.seed(83)
  xgb_tune <- xgb.cv(
    params = params,
    data = as.matrix(trainH_84[,-1]),
    label = trainH_84[,1],
    nrounds = 500,
    nfold = 10,
    objective = "reg:squarederror",
    verbose = 0,
    early_stopping_rounds = 10
  )
  hyper_grid$nrounds[i] <- which.min(xgb_tune$evaluation_log$test_rmse_mean)
  hyper_grid$RMSE[i] <- min(xgb_tune$evaluation_log$test_rmse_mean)
}

hyper_grid %>% arrange(RMSE) %>% head(10)

#build the XGBoost model with optimized parameters
set.seed(84)
xgb_84 <- xgboost(data = as.matrix(trainH_84[,-1]), label = trainH_84[,1],
                  eta = 0.05, max_depth = 7, 
                  min_child_weight = 5, 
                  subsample = 0.6, colsample_bytree = 0.7,
                  nrounds = 281,
                  objective = "reg:squarederror")

#predict CCS of testing set using xgb_84
pred_H_xgb_84 <- predict(xgb_84, as.matrix(testH_84[,-1]))
pred_H_xgb_84 <- data.frame(pred = exp(pred_H_xgb_84), empirical = exp(testH_84$ccs))
pred_H_xgb_84 <- pred_H_xgb_84 %>% mutate(error = (pred - empirical)/empirical*100)


rmsep <- RMSE(pred_H_xgb_84$pred, pred_H_xgb_84$empirical)
r2_test <-R2(pred_H_xgb_84$pred, pred_H_xgb_84$empirical)
sum(pred_H_xgb_84$error<2 & pred_H_xgb_84$error>-2)/length(pred_H_xgb_84$error)
sum(pred_H_xgb_84$error<3 & pred_H_xgb_84$error>-3)/length(pred_H_xgb_84$error)
sum(pred_H_xgb_84$error<5 & pred_H_xgb_84$error>-5)/length(pred_H_xgb_84$error)
median(abs(pred_H_xgb_206$error))

pred_H_xgb_84 <- pred_H_xgb_84 %>% mutate(data="CDK_84", algorithm = "XGBoost")
write.csv(pred_H_xgb_84, file = "pred_H_xgb_84.csv")

# calculate XGBoost importance
imp_var <- xgb.importance(feature_names = colnames(trainH_84)[-1], model = xgb_84)




#check the effect of halogen compounds...................................................
trainH_84_formula <- read.csv("trainH_84.csv", header = T, sep = ",")
testH_84_formula <- read.csv("testH_84.csv", header = T, sep = ",")

#excluding halogenated compounds from training set
trainH_84_nohalogen <- trainH_84_formula %>% filter(!grepl("Cl|Br|I|F", MolecularFormula))
rownames(trainH_84_nohalogen) <- trainH_84_nohalogen$name
trainH_84_nohalogen <- trainH_84_nohalogen[,-c(1,2)]

#separate halogenated and non-halogenated compounds in testing set
testH_84_halogen <- testH_84_formula %>% filter(grepl("Cl|Br|I|F", MolecularFormula))
rownames(testH_84_halogen) <- testH_84_halogen$name
testH_84_halogen <- testH_84_halogen[,-c(1,2)]

testH_84_nohalogen <- testH_84_formula %>% filter(!grepl("Cl|Br|I|F", MolecularFormula))
rownames(testH_84_nohalogen) <- testH_84_nohalogen$name
testH_84_nohalogen <- testH_84_nohalogen[,-c(1,2)]


#build SVM model with 84 descriptors with non-halogenated compounds
tune.svm(ccs ~., data = trainH_84_nohalogen, gamma = c(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/84, cost = 2^(-8:8))
svm_84_nohalogen <- svm(ccs~.,data=trainH_84_nohalogen, gamma = 0.000297619, cost = 256)

#predict CCS of 244 non-halogenated compounds in testing set
svm_pred_nohalogen <- predict(svm_84_nohalogen, testH_84_nohalogen[,-1])
rmsep <- RMSE(exp(svm_pred_nohalogen),exp(testH_84_nohalogen$ccs))
r2_test <-R2(exp(svm_pred_nohalogen),exp(testH_84_nohalogen$ccs))

pred_H_nohalogen <- data.frame(pred = exp(svm_pred_nohalogen), empirical = exp(testH_84_nohalogen$ccs))
pred_H_nohalogen <- pred_H_nohalogen %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_H_nohalogen$error<2 & pred_H_nohalogen$error>-2)/length(pred_H_nohalogen$error)
sum(pred_H_nohalogen$error<3 & pred_H_nohalogen$error>-3)/length(pred_H_nohalogen$error)
sum(pred_H_nohalogen$error<5 & pred_H_nohalogen$error>-5)/length(pred_H_nohalogen$error)
median(abs(pred_H_nohalogen$error))


#predict CCS of 85 halogenated compounds in testing set
svm_pred_halogen <- predict(svm_84_nohalogen, testH_84_halogen[,-1])
rmsep <- RMSE(exp(svm_pred_halogen),exp(testH_84_halogen$ccs))
r2_test <- R2(exp(svm_pred_halogen),exp(testH_84_halogen$ccs))

pred_H_halogen <- data.frame(pred = exp(svm_pred_halogen), empirical = exp(testH_84_halogen$ccs))
pred_H_halogen <- pred_H_halogen %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_H_halogen$error<2 & pred_H_halogen$error>-2)/length(pred_H_halogen$error)
sum(pred_H_halogen$error<3 & pred_H_halogen$error>-3)/length(pred_H_halogen$error)
sum(pred_H_halogen$error<5 & pred_H_halogen$error>-5)/length(pred_H_halogen$error)
median(abs(pred_H_halogen$error))




# building SVM model using the 15 descriptors from AllCCS
desc_allccs <- c("MW", "VAdjMat", "FMF", "nAtomLAC", "khs.ssCH2", 
                 "Kier3", "nAtom", "VP.0", "C2SP3", "VP.1", 
                 "MDEC.12", "ALogp2", "VP.2","nAtomLC","AMR")

scaled_H_allccs <- scaled_H[,c("ccs", desc_allccs)]
trainH_allccs <- scaled_H_allccs[rownames(trainH_84),]
testH_allccs <- scaled_H_allccs[rownames(testH_84),]


tune.svm(ccs ~., data = trainH_allccs, gamma = c(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)/15, cost = 2^(0:8))
svm_allccs <- svm(ccs~.,data = trainH_allccs, gamma = 0.003333333, cost = 256)


svm_pred_allccs <- predict(svm_allccs, testH_allccs[,-1])
rmsep <- RMSE(exp(svm_pred_allccs),exp(testH_allccs$ccs))
r2_test <- R2(exp(svm_pred_allccs),exp(testH_allccs$ccs))

pred_H_allccs <- data.frame(pred = exp(svm_pred_allccs), empirical = exp(testH_allccs$ccs))
pred_H_allccs <- pred_H_allccs %>% mutate(error = (pred - empirical)/empirical*100)
sum(pred_H_allccs$error<2 & pred_H_allccs$error>-2)/length(pred_H_allccs$error)
sum(pred_H_allccs$error<3 & pred_H_allccs$error>-3)/length(pred_H_allccs$error)
sum(pred_H_allccs$error<5 & pred_H_allccs$error>-5)/length(pred_H_allccs$error)
median(abs(pred_H_allccs$error))



# predict the CCS of unknowns, an example is shown below.

#first get the mean and sd of each descriptors
variable <- colnames(trainH_84)[-1]
H_84des <- H_84[,variable]
mean <- apply(H_84des, 2, mean)
sd <- apply(H_84des, 2, sd)

#then import and scale the descriptors of example by mean and sd obtained above
example <- read.csv("example.csv", header = T, sep = ",", row.names = "name")
example <- example[,variable]

example_center <- sweep(example, 2, mean, "-")
example_scaled <- sweep(example_center, 2, sd, "/")

#predict the CCS of example
pred_example <- predict(svm_84, example_scaled)
pred_example <- data.frame(pred = exp(pred_example))

write.csv(pred_example, file = "pred_example.csv")


















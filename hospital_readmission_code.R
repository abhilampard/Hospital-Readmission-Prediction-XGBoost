#Hospital Readmission Prediction
#Name: Abhishek Patil

setwd("C:/Users/Abhishek/Desktop/Hospital Readmission Prediction/Challenge")
options(repr.matrix.max.cols=50, repr.matrix.max.rows=100)
options(warn=-1)

#Libraries
library(data.table)
library(xgboost)
library(Matrix)
library(caret)
library(dummies)
library(pROC)

#--------------------------------------------- PART 1 ---------------------------------------------# 
train <- read.csv('challengetraining_data.csv')

#Data Preprocessing includes dropping columns, deleting some rows, changing column types from categorical to numeric, etc.

#Data Summary before processing. 
#Most of the preprocessing steps are based on the results of this summary.
summary(train)

#Defining a function for preprocessing
preprocessing <- function(train) 
  {
  #Dropping the ID columns
  train$encounter_id <- NULL
  train$patient_nbr <- NULL
  
  #Dealing with Special Characters (Replacing "?" with NA values)
  train[train == "?"] <- NA
  
  #Converting Race to numeric
  train$race <- as.numeric(as.factor(train$race))
  
  #Converting Age ranges into numeric values
  train$age <- ifelse(train$age == "[0-10)",  0, train$age)
  train$age <- ifelse(train$age == "[10-20)", 1, train$age)
  train$age <- ifelse(train$age == "[20-30)", 2, train$age)
  train$age <- ifelse(train$age == "[30-40)", 3, train$age)
  train$age <- ifelse(train$age == "[40-50)", 4, train$age)
  train$age <- ifelse(train$age == "[50-60)", 5, train$age)
  train$age <- ifelse(train$age == "[60-70)", 6, train$age)
  train$age <- ifelse(train$age == "[70-80)", 7, train$age)
  train$age <- ifelse(train$age == "[80-90)", 8, train$age)
  train$age <- ifelse(train$age == "[90-100)", 9, train$age)
  train$age <- as.numeric(train$age)
  
  #Converting Gender to numeric
  train <- train[!is.na(train$gender), ] #Dropping rows with NA values in Gender (2 rows of Unknown/Invalid)
  train$gender <- as.numeric(as.factor(train$gender))
  
  ##Converting Weight to numeric
  train$weight <- ifelse(train$weight == "[0-25)", 0, train$weight)
  train$weight <- ifelse(train$weight == "[25-50)", 1, train$weight)
  train$weight <- ifelse(train$weight == "[50-75)", 2, train$weight)
  train$weight <- ifelse(train$weight == "[75-100)", 3, train$weight)
  train$weight <- ifelse(train$weight == "[100-125)", 4, train$weight)
  train$weight <- ifelse(train$weight == "[125-150)", 5, train$weight)
  train$weight <- ifelse(train$weight == "[150-175)", 6, train$weight)
  train$weight <- ifelse(train$weight == "[175-200)", 7, train$weight)
  train$weight <- ifelse(train$weight == ">200", 8, train$weight)
  train$weight <- as.numeric(train$weight)
  
  #Converting the following columns to numeric/factors as applicable
  train$admission_type_id <- as.numeric(as.factor(train$admission_type_id))
  train$discharge_disposition_id <- as.numeric(as.factor(train$discharge_disposition_id))
  train$admission_source_id <- as.numeric(as.factor(train$admission_source_id))
  train$time_in_hospital <- as.numeric(train$time_in_hospital)
  train$payer_code <- as.numeric(as.factor(train$payer_code))
  train$medical_specialty <- as.numeric(as.factor(train$medical_specialty))
  train$num_lab_procedures <- as.numeric(train$num_lab_procedures)
  train$num_procedures <- as.numeric(train$num_procedures)
  train$num_medications <- as.numeric(train$num_medications)
  train$number_outpatient <- as.numeric(train$number_outpatient)
  train$number_emergency <- as.numeric(train$number_emergency)
  train$number_inpatient <- as.numeric(train$number_inpatient)
  train$diag_1 <- as.numeric(as.factor(train$diag_1))
  train$diag_2 <- as.numeric(as.factor(train$diag_2))
  train$diag_3 <- as.numeric(as.factor(train$diag_3))
  train$number_diagnoses <- as.numeric(train$number_diagnoses)
  
  #Converting max_glu_serum to numeric
  train$max_glu_serum <- ifelse(train$max_glu_serum == "None",  0, train$max_glu_serum)
  train$max_glu_serum <- ifelse(train$max_glu_serum == "Norm",  1, train$max_glu_serum)
  train$max_glu_serum <- ifelse(train$max_glu_serum == ">200",  2, train$max_glu_serum)
  train$max_glu_serum <- ifelse(train$max_glu_serum == ">300",  3, train$max_glu_serum)
  train$max_glu_serum <- as.numeric(train$max_glu_serum)
  
  #Converting A1Cresult to numeric
  train$A1Cresult <- ifelse(train$A1Cresult == "None",  0, train$A1Cresult)
  train$A1Cresult <- ifelse(train$A1Cresult == "Norm",  1, train$A1Cresult)
  train$A1Cresult <- ifelse(train$A1Cresult == ">7",    2, train$A1Cresult)
  train$A1Cresult <- ifelse(train$A1Cresult == ">8",    3, train$A1Cresult)
  train$A1Cresult <- as.numeric(train$A1Cresult);
  
  #Columns with over half of the data missing
  drops <- c("weight", "payer_code", "medical_specialty")
  train <- train[ , !(names(train) %in% drops)]
  
  #Columns having the same value throughout
  drops <- c("examide", "citoglipton")
  train <- train[ , !(names(train) %in% drops)]
  
  #Columns with very imbalanced categories
  drops <- c("chlorpropamide", "acetohexamide", "tolbutamide", "acarbose", "miglitol", "troglitazone", "tolazamide", "glipizide.metformin",
             "glimepiride.pioglitazone", "metformin.rosiglitazone", "metformin.pioglitazone", "nateglinide","glyburide.metformin")
  train <- train[ , !(names(train) %in% drops)]
  
  #Columns with Numeric and String values
  #Can be converted to numeric. Reference: (https://en.wikipedia.org/wiki/List_of_ICD-9_codes)
  #Due to limitation of time, dropping it.
  drops <- c("diag_1", "diag_2", "diag_3")  
  train <- train[ , !(names(train) %in% drops)]
  
  #Converting change to numeric
  train$change <- as.character(train$change)
  train$change [train$change  == "Ch"] <- 1
  train$change [train$change  == "No"] <- 0
  train$change  <- as.numeric(train$change)
  
  #Converting diabetesMed to numeric
  train$diabetesMed  <- as.character(train$diabetesMed)
  train$diabetesMed [train$diabetesMed  == "Yes"] <- 1
  train$diabetesMed [train$diabetesMed  == "No"] <- 0
  train$diabetesMed  <- as.numeric(train$diabetesMed)
  
  #Converting metformin, repaglinide, glimepiride, glipizide, glyburide, pioglitazone, rosiglitazone, insulin to numeric
  train$metformin <- as.character(train$metformin)
  train$repaglinide <- as.character(train$repaglinide)
  train$glimepiride <- as.character(train$glimepiride)
  train$glipizide <- as.character(train$glipizide)
  train$glyburide <- as.character(train$glyburide)
  train$pioglitazone <- as.character(train$pioglitazone)
  train$rosiglitazone <- as.character(train$rosiglitazone)
  train$insulin <- as.character(train$insulin)
  train[train == "Down"] <- -1
  train[train == "No"] <- 0
  train[train == "Steady"] <- 1
  train[train == "Up"] <- 2
  train$metformin <- as.integer(train$metformin)
  train$repaglinide <- as.numeric(train$repaglinide)
  train$glimepiride <- as.numeric(train$glimepiride)
  train$glipizide <- as.numeric(train$glipizide)
  train$glyburide <- as.numeric(train$glyburide)
  train$pioglitazone <- as.numeric(train$pioglitazone)
  train$rosiglitazone <- as.numeric(train$rosiglitazone)
  train$insulin <- as.numeric(train$insulin)
  
  return(train)
}

#Calling the defined function for data preprocessing
train <- preprocessing(train)

#Converting readmitted to numeric
train$readmitted <- as.character(train$readmitted)
train$readmitted[train$readmitted == "Y"] <- 1
train$readmitted[train$readmitted == "N"] <- 0
train$readmitted <- as.numeric(train$readmitted)

#Data Summary after processing
summary(train)

df <- train

#Train-Test Split 
set.seed(888)
train.index <- sample(nrow(df), nrow(df)*0.7)  
train.df <- df[train.index,]
valid.df <- df[-train.index,]

X_train <- train.df
X_test <- valid.df
y_train <- train.df$readmitted
y_test <- valid.df$readmitted

X_train$readmitted = NULL
X_test$readmitted = NULL

#### XGBoost Classifier ####
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
y_train <- as.matrix(y_train)
y_test <- as.matrix(y_test)

dtrain <- xgb.DMatrix(data = X_train,label = y_train)
dtest <- xgb.DMatrix(data = X_test,label=y_test)

#Since it is an imbalanced dataset, considering AUC as the evaluation metric.

params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  max_depth = 3,
  eta = 0.4,
  eval_metric = "auc"
)

xgbcv <- xgb.cv( params = params,
                 data = dtrain,
                 nrounds = 200,
                 nfold = 10,
                 stratified = T,
                 print_every_n = 20,
                 early_stopping_rounds = 10
)

xgb1 <- xgb.train (
  params = params, 
  data = dtrain, 
  watchlist = list(val=dtest,train=dtrain),
  print_every_n = 10, 
  nrounds = 200, 
  early_stopping_rounds = 10, 
  seed = 100
)

#Evaluation

#Training Accuracy
xgbpred_train <- predict (xgb1,dtrain)
#Threshold was set according to the accuracy score used
xgbpred_train <- ifelse (xgbpred_train > 0.12,1,0)
myroc <- roc(y_train, xgbpred_train)
cat("Training Accuracy: ", auc(myroc))


#Testing Accuracy
xgbpred_test <- predict (xgb1,dtest)

#Threshold was set according to the accuracy score used
xgbpred_test <- ifelse (xgbpred_test > 0.12,1,0)
myroc <- roc(y_test, xgbpred_test)
cat("Testing Accuracy: ", auc(myroc))


###### Final Model ######
#Train on the whole data 
X_train <- df
y_train <- df$readmitted

X_train$readmitted = NULL

X_train <- as.matrix(X_train)
y_train <- as.matrix(y_train)

dtrain_whole <- xgb.DMatrix(data = X_train,label = y_train)

xgbpred <- predict (xgb1, dtrain_whole)
#Threshold was set according to the accuracy score used
xgbpred <- ifelse (xgbpred > 0.12,1,0)
myroc <- roc(y_train, xgbpred)
cat("Final Model Accuracy: ", auc(myroc))

#Feature Importances
#mat <- xgb.importance (feature_names = colnames(X_train),model = xgb1)

#The plot shows the top 10 important features for this model. 
#xgb.plot.importance (importance_matrix = mat[1:15])

#Commenting the code for the plot as the markdown had problems displaying the plot
#The plot gives some interesting insights. Variables like number_inpatient, nu_lab_procedures, num_medication, time_in_hospital are important as one would have imagined.  

#--------------------------------------------- PART 2 ---------------------------------------------# 
#Prediction 

#Reading the test file
test <- read.csv('challengetest_data.csv')

#Creating a new dataframe for probabilities
predicted_probability <- data.frame("encounter_id" = test$encounter_id) 

#Calling the preprocessing function
test <- preprocessing(test)

#Creating a matrix for XGB
dtest_final <- xgb.DMatrix(data = as.matrix(test))

#Using the XGB model to predict probability
xgbpred_final_test <- predict (xgb1, dtest_final)

#Adding a column of probability to the new dataframe
predicted_probability$predicted_probability <- xgbpred_final_test

#Writing to a CSV file
write.csv(predicted_probability, file = "patil_abhishek.csv")

#The accuracy is not great but certainly better than a random guess. 

#Some of the things I would have loved to try out but couldn't due to limited time:
#1. EDA to visualize the patterns among the variables and their relationship with the dependent variable. 
#2. Correlation Plot Analysis, Hypothesis testing.
#3. Detailed Feature Engineering (Using Dummy Variables, dealing with missing values, etc.)
#4. Try out different models with Grid Search to compare performance.

#This was part of a challenge that was to be completed in 3 hours. Hence, this was just a preliminary investigation. 
#Any comments on what could be improved in this are appreciated.Thanks.
#--------------------------------------------- THE END ---------------------------------------------# 
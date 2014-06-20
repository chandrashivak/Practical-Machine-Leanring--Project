Project Report on Practical Machine Learning: 
========================================================

### Problem Statement:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. Source for the data is http://groupware.les.inf.puc-rio.br/har. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did exercise. This is the "classe" variable in the training set which needs to be predicted using other variables in the training set.

### Nature of the Data: 
This data has 19622 observations and each observation consists of 160 variables including the outcome variable "classe". The variable "classe" takes values from the set consisting of "A", "B", "C", "D" and "E". The variable "classe" needs to be predicted from other 159 variables. 

### Solution
The model to predict variable "classe" is on based random forest method and accuracy of this model is around 97%. Furthermore, our model predicted all 20 test cases correctly.  In the following, we explain all the steps involved in deriving the model. 

1. The entire data is split into two parts, the first part is 75% of total data for training and the second part is 25% of total data for cross validation. The cross validation data is to predict the accuracy of model. To split the data into two parts randomly, preProcess command with split factor of 0.75 is used.

2. We next removed the timestamps-related variables, the variable X and also the user names based on intuition because these parameters don't influence the outcome. This brings down the number of useful variables to 155.

3. Since 155 is a large number of variables to deal with, we first perform a NearZeroVar() operation to remove all the variables with zero or very low variances. In some sense, these variables do not provide much information. After this operation, we were able to bring down the variable count to 100.

4. The next important step is to perform a PCA analysis using the preProc() function. By setting a PCA threshold of 0.95, we were able to retain most of the variance with only 36 variables. These 36 variables form our 'predictors' and the classe variable is out 'outcome'. Given the density of NA variables in the data, we also used the "knnImpute" function to impute the data.

5. We used the train() function to train the data.
  * At first attempt, we only used a generalized linear model (glm) to fit the data. With this, we noticed that the prediction accuracy was as low as 55% on the cross-validation set.
  * Noting that the "glm" method works poorly for classification problems, we proceeded to use the random forest ("rf") technique of fitting data. In order to shorten the simulation time, we used the following control option on the training set: 'trControl=trainControl(method="cv",number= 2)'.  This provided a nice balance between simulation time and accuracy.
  
6. The performance of our algorithm as displayed by the confusionMatrix() function is presented below. As established by the accuracy metric, it does very well. The accuracy may be improved (slightly) using a higher PCA threshold and/or tweaking the train function control parameters.  

The summary of error analysis of cross-validation data is given below. As we can see below, the accuracy of our method is 97% and confidence interval for accuracy is between 96.6% and 97.5%. The following confusion matrix also provides sensitivity and specifity of each case. 
```
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1373    6    7    8    1
         B   13  915   17    2    2
         C    4   14  829    5    3
         D    4    1   33  759    7
         E    0    4    3    8  886

Overall Statistics
                                         
               Accuracy : 0.971          
                 95% CI : (0.966, 0.9756)
    No Information Rate : 0.2843         
    P-Value [Acc > NIR] : < 2.2e-16      
                                         
                  Kappa : 0.9634         
 Mcnemar's Test P-Value : 0.002002       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9849   0.9734   0.9325   0.9706   0.9855
Specificity            0.9937   0.9914   0.9935   0.9891   0.9963
Pos Pred Value         0.9842   0.9642   0.9696   0.9440   0.9834
Neg Pred Value         0.9940   0.9937   0.9852   0.9944   0.9968
Prevalence             0.2843   0.1917   0.1813   0.1595   0.1833
Detection Rate         0.2800   0.1866   0.1690   0.1548   0.1807
Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
Balanced Accuracy      0.9893   0.9824   0.9630   0.9798   0.9909

```
The detailed code used for prediction is given below.
```
library(caret)
library(kernlab)

# Loading the data
pmldata <- read.csv("pml-training.csv")
# Removing irrelevant variables such as name, time stamp, etc
pmldata <- pmldata[, -c(1:5)]

# Splitting data into two parts, one part is for training and another part is for cross-validation
set.seed(32323)
inTrain <- createDataPartition(y=pmldata$classe, p=0.75, list=FALSE)
training <- pmldata[inTrain, ]
testing <- pmldata[-inTrain, ]

# Eliminating near zero variance variables
nsv <- nearZeroVar(training[, ])
training_nzv <- training[, -nsv]

nameind <- which(names(training_nzv)=="user_name")
ocind <- which(names(training_nzv)=="classe")

# Applying PCA and removing NA with k neighbor impute method
preProc <- preProcess(training_nzv[, -c(nameind, ocind)], method=c("knnImpute", "pca"), thresh=0.95)
trainPC <- predict(preProc, training_nzv[, -c(nameind, ocind)])

# Applying the training based on random forest method
modelFit = train(training$classe~.,method="rf",data=trainPC,trControl=trainControl(method="cv",number= 2))

# Eliminating near zero variance variables, applying PCA and knn Impute on cross validatin data
testing_nzv <- testing[, -nsv]
testPC <- predict(preProc, testing_nzv[, -c(nameind, ocind)] )
testpredval <- predict(modelFit, testPC)
confusionMatrix(testing$classe, testpredval)

# Predicting the variable classe using test data
pml_testing <- read.csv("pml-testing.csv")
pml_testing <- pml_testing[, -c(1:5)]
pml_test_nzv <- pml_testing[, -nsv]
pml_testPC <- predict(preProc, pml_test_nzv[, -c(nameind, ocind)])
pml_testpredval <- predict(modelFit, pml_testPC)

```

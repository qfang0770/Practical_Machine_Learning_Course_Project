## Practical Machine Learning Course Project

### Qi Fang


### 1. Data loading

```r
setwd("~/Desktop/JHU/08_practicalMachineLearning")
rm(list = ls())
library(caret)
library(randomForest)
set.seed(1)
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
train <- read.csv(url(trainUrl), na.strings = c("NA", "#DIV/0!", ""))
test <- read.csv(url(testUrl), na.strings = c("NA", "#DIV/0!", ""))
```


### 2. Data partition
#### I use 60% of the "train" data as training set (ptrain) and 40% if the "train" data as validation set (pval).

```r
inTrain <- createDataPartition(y = train$classe, p = 0.6, list = FALSE)
ptrain <- train[inTrain, ]
pval <- train[-inTrain, ]
```


### 3. Data cleaning
#### Remove the variables with near zero variance.

```r
nzv <- nearZeroVar(ptrain, saveMetrics = TRUE)
ptrain <- ptrain[, nzv$nzv == FALSE]
pval <- pval[, nzv$nzv == FALSE]
```

#### Remove the variables with mostly NA values (>95% of the values are labeled with NA).

```r
mostlyNA <- sapply(ptrain, function(x) mean(is.na(x))) > 0.95
ptrain <- ptrain[, mostlyNA == FALSE]
pval <- pval[, mostlyNA == FALSE]
```

#### Remove the first five variables that are not relavent to the prediciton of classe.

```r
names(ptrain)[1:5]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"
```

```r
ptrain <- ptrain[, -(1:5)]
pval <- pval[, -(1:5)]
```

#### Note that I choose the features to be removed from only ptrain set, and the pval set is not seen, so the out-of-sample error will not be underestimated. Then I remove those features in both ptrain and pval set。


### 4. Model building
#### Here I build models to predict the classe from the predictors. The in-sample-error is estimated from the ptrain set and the out-sample-error is estimated from the pval set. The model with the smallest out-sample-error is selected.

#### First I fit a random forest model to ptrain set, and use 3-fold cross validation tp find the optimal tuning parameters.

```r
fitControl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
fit_rf <- train(classe ~ ., data = ptrain, method = "rf", trControl = fitControl)
fit_rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.31%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    5 2270    3    1    0 0.0039491005
## C    0    7 2045    2    0 0.0043816943
## D    0    0    8 1921    1 0.0046632124
## E    0    1    0    7 2157 0.0036951501
```
#### The in-sample-error of random forest model is 0.31%

#### Then I fit a gradient boosted model to ptrain set, and use 3-fold cross validation tp find the optimal tuning parameters.

```r
fitControl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
fit_boost <- train(classe ~ ., data = ptrain, method = "gbm",trControl = fitControl)
```

```r
fit_boost
```

```
## Stochastic Gradient Boosting 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 7852, 7850, 7850 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7579845  0.6929221  0.011314309
##   1                  100      0.8322874  0.7877069  0.007735681
##   1                  150      0.8705852  0.8361918  0.006649749
##   2                   50      0.8774633  0.8447482  0.004324310
##   2                  100      0.9356319  0.9185361  0.006393220
##   2                  150      0.9577956  0.9465989  0.003612193
##   3                   50      0.9307063  0.9122993  0.004672905
##   3                  100      0.9679852  0.9594947  0.003898674
##   3                  150      0.9829311  0.9784082  0.004005259
##   Kappa SD   
##   0.014359466
##   0.009763753
##   0.008414882
##   0.005497400
##   0.008106469
##   0.004566373
##   0.005946526
##   0.004933611
##   0.005067190
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```
#### The in-sample-error of gradient boosted model is 1 - 0.9829 = 1.71%. Both models show minimal error rates, and the random forest model is even better. Next, I will evaluate the two models on validation set to estimate the out-sample-error and select the best model.

### 5. Model evaluation and selection
#### I fit the models to validation set, and use the confusion matrix to examine the accuracy.

```r
## Random forest model
preds_rf <- predict(fit_rf, newdata = pval)
confusionMatrix(pval$classe, preds_rf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    1 1515    2    0    0
##          C    0    1 1364    3    0
##          D    0    0    6 1280    0
##          E    0    0    0    5 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9977          
##                  95% CI : (0.9964, 0.9986)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9971          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9993   0.9942   0.9938   1.0000
## Specificity            1.0000   0.9995   0.9994   0.9991   0.9992
## Pos Pred Value         1.0000   0.9980   0.9971   0.9953   0.9965
## Neg Pred Value         0.9998   0.9998   0.9988   0.9988   1.0000
## Prevalence             0.2846   0.1932   0.1749   0.1642   0.1832
## Detection Rate         0.2845   0.1931   0.1738   0.1631   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9998   0.9994   0.9968   0.9964   0.9996
```

```r
## Gradient boosted model
preds_boost <- predict(fit_boost, newdata = pval)
confusionMatrix(pval$classe, preds_boost)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227    3    0    2    0
##          B   14 1490   12    2    0
##          C    0    9 1356    3    0
##          D    0    2   11 1272    1
##          E    1    8    2   16 1415
## 
## Overall Statistics
##                                           
##                Accuracy : 0.989           
##                  95% CI : (0.9865, 0.9912)
##     No Information Rate : 0.2858          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9861          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9933   0.9854   0.9819   0.9822   0.9993
## Specificity            0.9991   0.9956   0.9981   0.9979   0.9958
## Pos Pred Value         0.9978   0.9816   0.9912   0.9891   0.9813
## Neg Pred Value         0.9973   0.9965   0.9961   0.9965   0.9998
## Prevalence             0.2858   0.1927   0.1760   0.1651   0.1805
## Detection Rate         0.2838   0.1899   0.1728   0.1621   0.1803
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9962   0.9905   0.9900   0.9901   0.9975
```
#### The accuracy of random forest model is 99.77%, and that of gradient boosted model 98.94%. Both model show excellent accuracy on the unseen validation set, and again random forest model is even better. Thus, in the following section I will fit the random forest model to the whole training set ("train") and predict the classes of testing set ("test").


### 6. Predicting classes of testing set using random forest model
#### Fit the random forest model to the whole training set ("train") and predict the classes of testing set ("test").

```r
nzv <- nearZeroVar(train)
train <- train[, -nzv]
test <- test[, -nzv]
mostlyNA <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[, mostlyNA == FALSE]
test <- test[, mostlyNA == FALSE]
train <- train[, -(1:5)]
test <- test[, -(1:5)]
fitControl <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
fit <- train(classe ~ ., data = train, method = "rf", trControl = fitControl)
preds <- predict(fit, newdata = test)
preds <- as.character(preds)
```

#### Finally I output the predictions.

```r
pml_write_files <- function(x) {
  n <- length(x)
  for(i in 1:n) {
    filename <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
  }
}
pml_write_files(preds)
```

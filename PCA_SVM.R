setwd("D:/Practice/R/Kaggle/Digits Recognizer/")
train = read.csv("train.csv")
test = read.csv("test.csv")
library(caTools)
library(randomForest)
library(rpart.plot)
library(rpart)
library(h2o)
library(readr)
library(ggplot2)
library(e1071)

label_train = train$label

split = sample.split(train$label,SplitRatio = 0.7)
tr = subset(train,split == T)
te = subset(train,split == F)

label_tr = tr$label
label_te = te$label

train$label = NULL
tr$label = NULL
te$label = NULL

combi = rbind(train,test)
pca.train = combi[1:nrow(train),]
pca.test = combi[-(1:nrow(train)),]
apply(train,2,var)

pca.tr = prcomp(tr)
par(mar = rep(2,4))
plot(pca.tr,type = "l")
pca.tr$rotation = -pca.tr$rotation
pca.tr$x = -pca.tr$x
biplot(pca.tr,scale = 0)
tr.data = predict(pca.tr,tr)
te.data = predict(pca.tr,te)
tr.data = data.frame(label = as.factor(label_tr),tr.data)
te.data = as.data.frame(te.data)
tr.data = tr.data[,1:50]
te.data = te.data[,1:50]

##SVM
SVM_tr = svm(label ~ .,tr.data)
SVM_tr.pred = predict(SVM_tr,te.data,method = "class")
head(SVM_tr.pred)
table(label_te,SVM_tr.pred)
(1229+1389+1220+1260+1199+1112+1224+1284+1185+1208)/12600
##COOl ENough

pca = prcomp(train)
pca.train.data = predict(pca,train)
pca.test.data = predict(pca,test)
pca.train.data = data.frame(label = as.factor(label_train),pca.train.data)
pca.test.data = as.data.frame(pca.test.data)
pca.train.data = pca.train.data[,1:50]
pca.test.data = pca.test.data[,1:50]

SVM = svm(label ~ ., pca.train.data)
summary(SVM)
SVM.pred = predict(SVM,newdata = pca.test.data)
head(SVM.pred)
prediction = data.frame(ImageId = 1:nrow(test),Label = SVM.pred)
head(prediction)
write.csv(prediction,"PCA_SVM.CSV",row.names = F)

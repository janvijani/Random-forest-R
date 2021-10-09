#My Random Forest
# multiple DT based on multiple bootstraps
HR_data<-read.csv(file.choose(), stringsAsFactors = T, 
                       na.strings = c("","", "NA"))
dim(HR_data)
colSums(is.na(HR_data))
summary(HR_data)
fix(HR_data)

# EmployeeCount,EmployeeNumber,Over18,StandardHours
HR_data$EmployeeCount<-NULL
HR_data$Over18<-NULL
HR_data$EmployeeNumber<-NULL
HR_data$StandardHours<-NULL

# convert required variables in factor 
#EnvironmentSatisfaction,JobInvolvement,JobLevel,JobSatisfaction,PerformanceRating,
#RelationshipSatisfaction,StockOptionLevel,WorkLifeBalance


vect1<-c("EnvironmentSatisfaction","JobInvolvement","JobLevel","JobSatisfaction",
         "PerformanceRating",
         "RelationshipSatisfaction","StockOptionLevel","WorkLifeBalance")

HR_data[,vect1]<-lapply(HR_data[,vect1],as.factor)
summary(HR_data)

quantile(HR_data$YearsAtCompany,probs = seq(0.9,1.0,0.01))
HR_data$YearsAtCompany[HR_data$YearsAtCompany>15]<-15

HR_data$TrainingTimesLastYear<-as.factor(HR_data$TrainingTimesLastYear)

table(HR_data$Attrition)
# No  Yes 
#1233  237
install.packages("randomForest")
library(randomForest)

rf_attrition<-randomForest(Attrition~.,data = HR_data)
rf_attrition
floor(sqrt(ncol(HR_data)))
#No Yes class.error
#No  1222  11  0.00892133
#Yes  198  39  0.83544304

#this means we can predict NOs correctly but the error rate of wrongly Identifies YESs is 83% 
#these are out of banks error 

plot(rf_attrition)


#now we need to modify our database and perform up/down sampling to get bettwe results 

install.packages("caret")
library(caret)

up_sample<-upSample(HR_data,HR_data$Attrition)
dim(up_sample)
table(up_sample$Attrition)
#No   Yes 
#1233 1233 

up_sample$Class<-NULL
#we need the above code because models create an extra class column which will not hinder outcomes 
rf_attrition_upsample<-randomForest(Attrition~.,data = up_sample)
rf_attrition_upsample

#OOB estimate of  error rate: 1.09%
#Confusion matrix:
#  No  Yes class.error
#No  1208   25  0.02027575
#Yes    2 1231  0.00162206

#now we got excellent accuracy 
#but this is an unbiased data that we created. So lets try how accurately it predicts as compared to real data 


predicted_attrition <-predict(rf_attrition_upsample,HR_data)
head(predicted_attrition)
table(actual=HR_data$Attrition,predicted=predicted_attrition)
#predicted
#actual   No  Yes
#No  1233    0
#Yes    0  237

#100% accuracy yayyy

# we will take up - sample data

# we will divide that data in to 75% and 25% data 
set.seed(200)
index<-sample(nrow(up_sample),0.75*nrow(up_sample))
nrow(up_sample) # 2466
#index<-sample(2466,0.75*2466)
training_hr<-up_sample[index,]
testing_hr<-up_sample[-index,]
dim(training_hr)
dim(testing_hr)

# buiild model on training data 

# we will build model on 75% of data 
rf_attrition_train<-randomForest(Attrition~.,data = training_hr)
pred_attrition_train<-predict(rf_attrition_train,training_hr)
table(actual=training_hr$Attrition,predicted=pred_attrition_train)


# we will try to implement that model on 25% of the data 
pred_attrition_test<-predict(rf_attrition_upsample,testing_hr)
table(actual=testing_hr$Attrition,predicted=pred_attrition_test)
#100% accuracy 

#now lets perform binary logistic regression on it 

#####################################BLR#################################

#implementing blr on training data
summary(training_hr)
#creating null model and full model 

null_hr_model<-glm(Attrition~1,data = training_hr,family = "binomial")
full_hr_model<-glm(Attrition~.,data = training_hr,family = "binomial")

#now we perform stepwise regression , this time lets go with backward kind of stepwise regression 

step(full_hr_model,direction = "backward",
     scope = list(lower=null_hr_model,upper=full_hr_model))

hr_model_bin<-glm(formula = Attrition ~ Age + BusinessTravel + DailyRate + 
                    Department + DistanceFromHome + EducationField + EnvironmentSatisfaction + 
                    Gender + JobInvolvement + JobLevel + JobRole + JobSatisfaction + 
                    MaritalStatus + MonthlyIncome + NumCompaniesWorked + OverTime + 
                    RelationshipSatisfaction + StockOptionLevel + TotalWorkingYears + 
                    TrainingTimesLastYear + WorkLifeBalance + YearsAtCompany + 
                    YearsInCurrentRole + YearsSinceLastPromotion + YearsWithCurrManager, 
                  family = "binomial", data = training_hr)

# predict probability of churn using glm
pred_prob_attrition_train_bin<-predict(hr_model_bin,training_hr,type="response")
# if prob > 0.5 , convert as "Yes" else "No"

pred_attrition_train_bin<-ifelse(pred_prob_attrition_train_bin>=0.5,"Yes","No")

table(actual=training_hr$Attrition,predicted=pred_attrition_train_bin)

# accuracy , recall_Yes , recall_No
acc_train<-(740+777)/nrow(training_hr);acc_train   #0.82
recall_yes<-777/(145+777);recall_yes #0.84
recall_no<-740/(740+187);recall_no #0.79

#implementing blr on test data

pred_prob_attrition_test<-predict(hr_model_bin,testing_hr,type="response")
pred_attrition_test<-ifelse(pred_prob_attrition_test>=0.5,"Yes","No")
table(actual=testing_hr$Attrition,predicted=pred_attrition_test)
acc_test<-(246+249)/nrow(testing_hr);acc_test #0.80
recall_yes_test<-246/(60+246);recall_yes_test #0.80
recall_no_test<-249/(249+62);recall_no_test #0.80

# using the binary model " hr_model_bin" try to predict results on original data
# HR_data 

pred_prob_HR_data<-predict(hr_model_bin,HR_data,type = "response")
pred_attrition_HR<-ifelse(pred_prob_HR_data>=0.5,"Yes","No")
table(actual=HR_data$Attrition,predicted=pred_attrition_HR)
acc_HR<-(986+195)/nrow(HR_data);acc_HR #0.80
recall_Yes_HR<-195/(42+195);recall_Yes_HR #0.8270042
recall_No_HR<-986/(986+247);recall_No_HR #0.79

###################################
# sapply , lapply , tapply

df<-cars
View(df)

a<-sapply(df,max)
class(a)
# output is vector 
#we use sapply function if the data is homogeneous 

b<-unlist(lapply(df,max))
b
class(b)
# output is list
#we use sapply function if the data is heterogeneous 

df<-iris
head(df)
View(df)
# var of every column using sapply and lapply 
sapply(df[,1:4],var)
lapply(df[,1:4],var)

a<-tapply(df$Sepal.Length,df$Species,mean)
b<-aggregate(cbind(Sepal.Length,Sepal.Wi~Species,data = df,FUN = mean))
class(a)
class(b)

head(iris)
tail(cars)

###Clear The Buffer By Removing Pre-Built Variables#######

rm(list=ls())

############ Loading Packages##########
#Load Libraries
x = c("ggplot2", "corrgram",'DMwR','labeling')

lapply(x, require, character.only = TRUE)
rm(x)

#install.packages("scater","ggsignif","tidyverse","cluster","factoextra")

library(ggsignif)
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

####### Set Working Directory ######

rm(list=ls(all=T))
setwd("C:/Users/Sanjeev/Desktop/Edwisor/credit_card_project")

## Read the data
credit_a <- read.csv("credit-card-data.csv",header = TRUE)

#Removing the Cust_ID Column as It Is Categorical Variable and not useful for our analysis (Non-Numeric)
credit <- credit_a[,-1]

View(credit_a)
dim(credit)
colnames(credit)
str(credit)
####################################
di <- describe(credit)
di
############ Find the range of frequency variable in dataset #############
range(credit$BALANCE_FREQUENCY)#0-1
range(credit$PURCHASES_FREQUENCY)#0-1
range(credit$ONEOFF_PURCHASES_FREQUENCY)#0-1
range(credit$PURCHASES_INSTALLMENTS_FREQUENCY)#0-1
range(credit$CASH_ADVANCE_FREQUENCY)

########## Replace the freq >1 with 1 ############
credit$CASH_ADVANCE_FREQUENCY[credit$CASH_ADVANCE_FREQUENCY > 1] <- 1

library(Hmisc)
di <- describe(credit)
di

##################################Missing Values Analysis###############################################
missing_val = data.frame(apply(credit,2,function(x){sum(is.na(x))}))

missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(credit)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "Miising_perc.csv", row.names = F)

####################################################################################

ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
   geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
   ggtitle("Missing data percentage (Train)") + theme_bw()


############################### Missing value Impuatation ################################

#Mean Method
#credit$MINIMUM_PAYMENTS[is.na(credit$MINIMUM_PAYMENTS)] = mean(credit$MINIMUM_PAYMENTS, na.rm = T)
#credit$CREDIT_LIMIT[is.na(credit$CREDIT_LIMIT)] = mean(credit$CREDIT_LIMIT, na.rm = T)

#Median Method
#credit$MINIMUM_PAYMENTS[is.na(credit$MINIMUM_PAYMENTS)] = median(credit$MINIMUM_PAYMENTS, na.rm = T)
#credit$CREDIT_LIMIT[is.na(credit$CREDIT_LIMIT)] = median(credit$CREDIT_LIMIT, na.rm = T)


# kNN Imputation
credit = knnImputation(credit, k = 4)

#Check Missing value after KNN Imputation
missing_val_Check = data.frame(apply(credit,2,function(x){sum(is.na(x))}))
missing_val_Check$Columns = row.names(missing_val_Check)
names(missing_val_Check)[1] =  "Sum_of_Missing_Values"
missing_val_Check
###############################################################

#Taking a copy of credit
cc2=credit
# Box plot to analyze the outliers in given dataset
boxplot(cc2[,1:5],par(cex.lab=0.5))
boxplot(cc2[,6:11],par(cex.lab=0.5))
boxplot(cc2[,12:17],par(cex.lab=0.5))

#derived KPI's variables

cc2$monthly_avg_purchase <- cc2$PURCHASES/cc2$TENURE
cc2$monthly_cash_advance <- cc2$CASH_ADVANCE/cc2$TENURE      #how do i do it with mutate fn at once?
cc2$limit_usage <- cc2$BALANCE/cc2$CREDIT_LIMIT

cc2$purchase_type <- ifelse(cc2$ONEOFF_PURCHASES==0 & cc2$INSTALLMENTS_PURCHASES==0,'NONE',
                            ifelse(cc2$ONEOFF_PURCHASES>0 & cc2$INSTALLMENTS_PURCHASES==0,'one_off',
                                   ifelse(cc2$ONEOFF_PURCHASES==0 & cc2$INSTALLMENTS_PURCHASES>0,'installment',
                                          ifelse(cc2$ONEOFF_PURCHASES>0 & cc2$INSTALLMENTS_PURCHASES>0,'both','NA'))))
cc2$purchase_type_none <- ifelse(cc2$purchase_type=='NONE',1,0)
cc2$purchase_type_one_off <- ifelse(cc2$purchase_type=='one_off',1,0)
cc2$purchase_type_installment <- ifelse(cc2$purchase_type=='installment',1,0)
cc2$purchase_type_both <- ifelse(cc2$purchase_type=='both',1,0)


cc2$payment_minpayment <- cc2$PAYMENTS/cc2$MINIMUM_PAYMENTS    
cc2$TENURE <- as.numeric(cc2$TENURE)

####### droping Purchase type (categorical) avriable ########

cc2 <- cc2[,-21]

#----------Take a copy of cc2-------
cre_original <- cc2
##############
####  Extreme value Treatment
####  Since there are variables having extreme values so I am doing 
####  log-transformation on the dataset to remove outlier effect 

cc2_log = (log(cc2 + 1 ))
#---------- Deleting the features used in deriving KPI's 

cc2_subset = subset(cc2_log, select = -c(BALANCE,PURCHASES,CASH_ADVANCE,TENURE,
                                     PAYMENTS,MINIMUM_PAYMENTS,CREDIT_LIMIT))

#---------------------------------------------Feature selection--------------------------------
corrm <- cor(cc2_subset)

corrgram( cc2_subset, order = F,
          upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#-----------Correlation Plot---------------------

cr_dummy=subset(cc2_subset, select = -c(ONEOFF_PURCHASES_FREQUENCY,PURCHASES_INSTALLMENTS_FREQUENCY,
                     CASH_ADVANCE_FREQUENCY,PURCHASES_FREQUENCY,PURCHASES_TRX,CASH_ADVANCE_TRX))
#-------Shape of cr_dummy

dim(cr_dummy)

#------ cr_dummy contains all the variables that will be used for clustering

Scaled_data<- data.frame(scale(cr_dummy))

#------- Performing PCA

Scaled_data_pca <- prcomp(Scaled_data, center = TRUE)

summary(Scaled_data_pca)


#-------Scree Plot---------------------

plot(Scaled_data_pca)
screeplot(Scaled_data_pca, type='lines')


#--------Get the loadings-----------------

loadings_df <- data.frame(pc$loadings[,1:5])

#--------Get the first  principal component for further analysis-----------
Credit_PCs <- data.frame(Scaled_data_pca$x[,1:5])

#--------Get the dimension of Credit_PCs------------
dim(Credit_PCs)

#------Elbow Method for finding the optimal number of clusters-------
#------Plot Total within-clusters sum of squares Plot----------

set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- Credit_PCs
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 100 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#############################################

# function to compute total within-cluster sum of squares
#fviz_nbclust(Credit_PCs, kmeans, method = "wss", k.max = 24) + theme_minimal() + ggtitle("the Elbow Method")
   
# Silhouette method
fviz_nbclust(Credit_PCs, kmeans, method = "silhouette", k.max = 10) 

#building clusters using k-means clustering
CreditCluster <- kmeans(Credit_PCs, 4, nstart = 20)
CreditCluster

###Key performance variable selection . here I am taking variables which we will use in deriving new KPI.
#We can take all 17 variables but it will be difficult to interpret. So, we are selecting less no of variables.

cre_original= subset(cre_original,select= c(PURCHASES_TRX,ONEOFF_PURCHASES,INSTALLMENTS_PURCHASES,monthly_avg_purchase,monthly_cash_advance,limit_usage,CASH_ADVANCE_TRX,
         payment_minpayment,purchase_type_both,purchase_type_installment,purchase_type_none,purchase_type_one_off,CREDIT_LIMIT))

# Conactenating labels found through Kmeans with dataset as column 'Cluster'
cre_original$cluster <- as.factor(CreditCluster$cluster)

#------find the number of customers in each cluster---------
as.data.frame(table(CreditCluster$cluster))

# Mean value gives a good indication of the distribution of data. 
# So we are finding mean value for each variable for each cluster

cluster_4<-as.data.frame(aggregate( .~ cluster, FUN=mean, data=cre_original))
# transpose
cluster_4 <- t(cluster_4)

#!/usr/bin/env python
# coding: utf-8

# ### Goal of Model:

# ***Advanced data preparation:*** Build an ‘enriched’ customer profile by deriving “intelligent” KPIs such as:
#     
#     1- Monthly average purchase and cash advance amount
#     
#     2- Purchases by type (one-off, installments)
#     
#     3- Average amount per purchase and cash advance transaction,
#     
#     4- Limit usage (balance to credit limit ratio),
#     
#     5- Payments to minimum payments ratio etc.
#     
#     6- Advanced reporting: Use the derived KPIs to gain insight on the customer profiles.
#     
#     7- Identification of the relationships/ affinities between services.
#     
#     8- Clustering: Apply a data reduction technique factor analysis/PCA for variable reduction technique and a clustering algorithm to reveal the behavioural segments of credit card holders
#     
#     9- Identify cluster characterisitics.
#     
#     10- Provide the strategic insights and implementation of strategies for given set of cluster characterist

import pandas as pd
import numpy as np
import os
from fancyimpute import KNN

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sn


# In[228]:


#Set working directory
os.chdir("C:\\Users\\Sanjeev\\Desktop\\Edwisor\\credit_card_project")


# ### Load data

# In[229]:


credit= pd.read_csv("credit-card-data.csv")


# ### Information about data set

# In[230]:


credit.head()


# In[231]:


credit.shape


# In[232]:


credit.describe()


# # Data Exploration and Cleaning
# 
# Replace the frequency > 1 with 1

# In[233]:


credit['CASH_ADVANCE_FREQUENCY'].values[credit['CASH_ADVANCE_FREQUENCY'].values > 1] = 1


# In[234]:


# Find missing value in each feature
missing_val = credit.isnull().sum().sort_values(ascending=False)
#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_Value_count'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_Value_count']/len(credit))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
missing_val


# ---
# 
# ### a) Missing Value Treatment
#        - Since there are missing values in the data so we have to impute the missing value.
#        - We have three option to impute the missing value (mean, median, KNN imputation)
#        - To select the best method to impute the missing value 
# ---

# In[235]:


# Remove CUST_ID (not usefull)
credit.drop("CUST_ID", axis=1, inplace=True)


# In[236]:


# credit['MINIMUM_PAYMENTS'].loc[500]
# Actual = 457.255
# Mean = 1199.94
# Median = 873.09
# KNN Imputation (K=4) = 469.99


# In[237]:


print(credit['MINIMUM_PAYMENTS'].loc[500])


# In[238]:


credit['MINIMUM_PAYMENTS'].loc[500]=np.nan


# In[239]:


#Impute with mean
#credit = credit.fillna(credit.mean())

#Impute with median
#credit = credit.fillna(credit.median())

#Apply KNN imputation algorithm
credit = pd.DataFrame(KNN(k = 4).fit_transform(credit), columns = credit.columns)


# In[240]:


# Check missing value after imputing with KNN imputation 
# Find missing value in each feature
missing_val = credit.isnull().sum().sort_values(ascending=False)
#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_Value_count'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_Value_count']/len(credit))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
missing_val


# ***EXPLORATORY DATA ANALYSIS

# In[241]:


# Exploratory Data Analysis
credit.hist(figsize=(18,18))


# # Box plot to check the outlier in dataset

# In[242]:


#let´s see how are distributed the frequency variables

credit[['BALANCE_FREQUENCY',
 'PURCHASES_FREQUENCY',
 'ONEOFF_PURCHASES_FREQUENCY',
 'PURCHASES_INSTALLMENTS_FREQUENCY',
 'CASH_ADVANCE_FREQUENCY',
'PRC_FULL_PAYMENT']].plot.box(figsize=(18,10),title='Frequency',legend=True)
plt.tight_layout()


# In[243]:


#let´s see how are distributed the numeric variables

credit[['BALANCE',
 'PURCHASES',
 'ONEOFF_PURCHASES',
 'INSTALLMENTS_PURCHASES',
 'CASH_ADVANCE',
 'CREDIT_LIMIT',
 'PAYMENTS',
 'MINIMUM_PAYMENTS'
]].plot.box(figsize=(18,10),title='Distribution',legend=True)
plt.tight_layout()

# There are also many outliers, but we will keep them for now


# In[244]:


#let´s see how are distributed the numeric variables

credit[[ 'CASH_ADVANCE_TRX',
 'PURCHASES_TRX'
]].plot.box(figsize=(18,10),title='Distribution of transactions',legend=True)
plt.tight_layout()


# ### Deriving New KPI

# ***1. Monthly average purchase and cash advance amount***

# #### Monthly_avg_purchase

# In[245]:


credit['Monthly_avg_purchase']=credit['PURCHASES']/credit['TENURE']


# #### Monthly_cash_advance Amount

# In[246]:


credit['Monthly_cash_advance']=credit['CASH_ADVANCE']/credit['TENURE']


# ####  Purchases by type (one-off, installments)
# 
# - To find what type of purchases customers are making on credit card

# In[247]:


credit.loc[:,['ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES']]


# #### Find customers ONEOFF_PURCHASES and INSTALLMENTS_PURCHASES details

# In[248]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[249]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# In[250]:


credit[(credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0)].shape


# In[251]:


credit[(credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0)].shape


# ***As per above detail we found out that there are 4 types of purchase behaviour in the data set. So we need to derive a categorical variable based on their behaviour***

# In[252]:


def purchase(credit):
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (credit['ONEOFF_PURCHASES']>0) & (credit['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (credit['ONEOFF_PURCHASES']==0) & (credit['INSTALLMENTS_PURCHASES']>0):
        return 'istallment'


# In[253]:


credit['purchase_type']=credit.apply(purchase,axis=1)


# In[254]:


credit['purchase_type'].value_counts()


# #### Limit_usage (balance to credit limit ratio ) credit card utilization
#    - Lower value implies cutomers are maintaing thier balance properly. Lower value means good credit score

# In[255]:


credit['limit_usage']=credit.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)


# ####  Payments to minimum payments ratio 

# In[256]:


#PAYMENT_MINPAYMENT
#The where clause is being used to avoid div by zero error 
credit['payment_minpayment'] = np.where(credit['MINIMUM_PAYMENTS']== 0, credit['PAYMENTS'], 
                                        credit['PAYMENTS']/credit['MINIMUM_PAYMENTS'])


# In[257]:


credit['payment_minpayment'].describe()


# In[258]:


credit.shape


# 
# ####  Avg Amount per cash advance transaction is equivalent to CASH_ADVANCE_TRX 
# ####  Avg Amount per purchase is equivalent to AVERAGE_PURCHASE_TRX
# 

# 

# # Getting Insights from newly dervived KPI's

# #### Average payment_minpayment ratio for each purchse type.

# In[259]:


credit.groupby('purchase_type').apply(lambda x: np.mean(x['payment_minpayment'])).plot.bar()
plt.ylabel('payment-minimumpayement ratio')
plt.show()


# Insights : Customers with installments have highest payment-minimum payement ratio

# In[260]:


credit.groupby('purchase_type').apply(lambda x: np.mean(x['Monthly_avg_purchase'])).plot.bar()
plt.ylabel('MONTHLY_AVG_PURCHASE')
plt.show()


# Insights : Customers with one off and installments do most monthly purchase

# In[261]:


credit.groupby('purchase_type').apply(lambda x: np.mean(x['Monthly_cash_advance'])).plot.bar()
plt.ylabel('MONTHLY_AVG_CASH_ADVANCE')
plt.show()


# Insights : Customers with no one off and installments take more monthly cash advance

# In[262]:


credit.groupby('purchase_type').apply(lambda x: np.mean(x['limit_usage'])).plot.bar()
plt.ylabel('LIMIT_USAGE')
plt.show()


# Insights : Customers with no one off and installments have highest limit usage

# # Insights
# - Customers with installments have highest payment-minimum payement ratio
# - Customers with one off and installments do most monthly purchases
# - Customers with no one off and installments take more monthly cash advance
# - Customers with no one off and installments have highest limit usage

# ####  Extreme value Treatment
# - Since there are variables having extreme values so I am doing log-transformation on the dataset to remove outlier effect and make the data normally distributed 

# In[263]:


cr_log=credit.drop(['purchase_type'],axis=1).applymap(lambda x: np.log(x+1))


# In[264]:


cr_log.head()


# In[265]:


cr_log.describe()


# In[266]:


cr_log.head()


# In[267]:


col=['BALANCE','PURCHASES','CASH_ADVANCE','TENURE','PAYMENTS','MINIMUM_PAYMENTS','CREDIT_LIMIT']
cr_pre=cr_log[[x for x in cr_log.columns if x not in col ]]


# #### Original dataset with categorical column converted to number type.

# ***We do have some categorical data which need to convert with the help of dummy creation***

# In[268]:


cre_original=pd.concat([credit,pd.get_dummies(credit['purchase_type'])],axis=1)


# In[269]:


cre_original.head()


# In[270]:


cr_pre['purchase_type']=credit.loc[:,'purchase_type']
pd.get_dummies(cr_pre['purchase_type'], prefix='purchase_type')


# In[271]:


cr_dummy=pd.concat([cr_pre,pd.get_dummies(cr_pre['purchase_type'], prefix='purchase_type')],axis=1)


# In[272]:


cr_dummy=cr_dummy.drop('purchase_type',axis=1)
cr_dummy.isnull().any()


#  2. Data Manipulation

# #### B. Checking for multicollinearity

# In[273]:


##Correlation analysis
#Correlation plot
#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
credit_new_corr=cr_dummy.corr()

#Plot using seaborn library
sns.heatmap(credit_new_corr, mask=np.zeros_like(credit_new_corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[274]:


col=['ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','PURCHASES_FREQUENCY','PURCHASES_TRX','CASH_ADVANCE_TRX']
cr_dummy=cr_dummy[[x for x in cr_dummy.columns if x not in col ]]


# In[275]:


cr_dummy=pd.DataFrame(cr_dummy)
cr_dummy.shape


# ### C. Standardizing the data

# In[276]:


from sklearn.preprocessing import StandardScaler


# In[277]:


sc=StandardScaler()


# In[278]:


cr_scaled = sc.fit_transform(cr_dummy)


# In[279]:


cnames=cr_dummy.columns


# In[280]:


cr_scaled_df=pd.DataFrame(cr_scaled,columns=cnames)
cr_scaled_df.head()


# ### Applying PCA
# 
# **With the help of principal component analysis we will reduce features**

# In[281]:


from sklearn.decomposition import PCA


# In[282]:


cr_scaled_df.head()


# In[283]:


#We have 17 features so our n_component will be 132
pc=PCA(n_components=12)
cr_pca=pc.fit(cr_scaled)


# In[284]:


#Lets check if we will take 12 component then how much varience it explain. Ideally it should be 1 i.e 100%
sum(cr_pca.explained_variance_ratio_)


# In[285]:


var_ratio={}
for n in range(2,13):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(cr_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)


# In[286]:


var_ratio


# #### Performing Factor Analysis

# In[287]:


pd.Series(var_ratio).plot()


# In[288]:


pc_final=PCA(n_components=5).fit(cr_scaled)

reduced_cr=pc_final.fit_transform(cr_scaled)


# In[289]:


dd=pd.DataFrame(reduced_cr)
reduced_cr.shape


# In[290]:


dd.head()


# In[ ]:





# ***So initially we had 17 variables now its 5 so our variable go reduced***

# In[291]:


pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(5)],index=cnames)


# In[292]:


# Factor Analysis : variance explained by each component- 
pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(5)])


# ### Cluster Analysis

# ### Choosing number clusters using Pseudo F-value (elbow method)

# In[293]:


#Load required libraries
from sklearn.cluster import KMeans

#Estimate optimum number of clusters
cluster_range = range( 1, 20 )
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans(num_clusters).fit(reduced_cr)
    cluster_errors.append(clusters.inertia_)
    
#Create dataframe with cluster errors
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )


# In[294]:


#Plot line chart to visualise number of clusters
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# ### Note:
# - The elbow diagram shows that the gain in explained variance reduces significantly from 3 to 4. So, optimal number of clusters could be 4. 

# ### Choosing number clusters using Silhouette Coefficient (SC)

# In[295]:


from sklearn import metrics


# In[296]:


# calculate SC for K=2 to K=10
k_range = range(2, 10)
scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=100)
    km.fit(reduced_cr)
    scores.append(metrics.silhouette_score(reduced_cr, km.labels_))


# In[297]:


# plot the results
plt.plot(k_range, scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)


# ##### Note
# 
# The solution can be 4 based on the SC score. If we take highest SC score, 4 segment solution is best

# ### CALCULATING K-MEANS AND THE CENTROIDS

# In[298]:


# It seems that the optimal number of clusters is 4.
# I am going to take 4 for the analysis
km_4=KMeans(n_clusters=4,random_state=123)


# In[299]:


# applying kmeans
km_4.fit(reduced_cr)


# In[300]:


pd.Series(km_4.labels_).value_counts()


# ### ADDING THE LABELS TO THE DATASET

# In[301]:


df_pair_plot=pd.DataFrame(reduced_cr,columns=['PC_' +str(i) for i in range(5)])


# In[302]:


df_pair_plot['Cluster']=km_4.labels_ 


# In[303]:


df_pair_plot.head()


# In[304]:


color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in km_4.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=label_color,cmap='Spectral',alpha=0.1)


# PC0 and PC1 are able to distinguish the cluster clearly

# In[305]:


#pairwise relationship of components on the data
sns.pairplot(df_pair_plot,hue='Cluster', palette= 'Dark2', diag_kind='kde',size=1.85)


# ### CLUSTERS EXPLANATION AND MARKETING STRATEGY

# In[306]:


#Key performance variable selection . here I am taking variables which we will use in deriving new KPI.
#We can take all 17 variables but it will be difficult to interpret. So, we are selecting less no of variables.

col_kpi=['PURCHASES_TRX','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES','Monthly_avg_purchase','Monthly_cash_advance','limit_usage','CASH_ADVANCE_TRX',
         'payment_minpayment','both_oneoff_installment','istallment','one_off','none','CREDIT_LIMIT']


# In[307]:


cre_original.describe()


# In[308]:


# Conactenating labels found through Kmeans with data 
cluster_df_4=pd.concat([cre_original[col_kpi],pd.Series(km_4.labels_,name='Cluster_4')],axis=1)


# In[309]:


cluster_df_4.head()


# In[310]:


# Mean value gives a good indication of the distribution of data. 
#So we are finding mean value for each variable for each cluster
cluster_4=cluster_df_4.groupby('Cluster_4').apply(lambda x: x[col_kpi].mean()).T
cluster_4


# In[311]:


fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(cluster_4.columns))

cash_advance=np.log(cluster_4.loc['Monthly_cash_advance',:].values+1)
credit_score=(cluster_4.loc['limit_usage',:].values+1)
purchase= np.log(cluster_4.loc['Monthly_avg_purchase',:].values+1)
payment=np.log(cluster_4.loc['payment_minpayment',:].values+1)
installment=cluster_4.loc['istallment',:].values
one_off=cluster_4.loc['one_off',:].values
both=cluster_4.loc['both_oneoff_installment',:].values



bar_width=.08
b1=plt.bar(index,cash_advance,color='b',label='Monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='Credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='Avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='Payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='One_off',width=bar_width)
b6=plt.bar(index+5*bar_width,both,color='y',label='both_oneoff_installment',width=bar_width)


plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()


# ***Insights with 4 Clusters***
# 
# 
# - Cluster 0 customers are doing maximum Installments transactions, take least monthly cash advance, poor credit_score and highest payment minpayment ratio ***This group is about 25% of the total customer base***
# 
# 
# - cluster 1 is taking maximum advance_cash  and   is paying comparatively less minimum payment and good credit_score & doing no purchase transaction. ***This group is about 23% of the total customer base***
# 
# 
# - Cluster 2 is the group of customers who are doing maximum one off transactions, have comparatively good credit score and lowest payment minpayment ratio. ***This group is about 21% of the total customer base ***
# 
# 
# - Cluster 3 customers who are doing both one off and Instllment transactions have good credit score and do maximum monthly purchase.*** This group is about 31% of the total customer base***

# ### Marketing Strategy Suggested:
# 
# #### a. Group 0
#    - They are potential target customers who are paying dues and doing mostly installment purchases and have poor credit score)
#        --    we can increase credit limit or can lower down interest rate
#        --    Can be given premium card /loyality cards to increase transactions
#        
# #### b. Group 1
#    - They have good credit score and taking only cash on advance. We can target them by providing less interest rate on Installment purchase transaction or cashback/discount on one off purchase transaction.
#    
# #### c. Group 2
#    - This group is has minimum paying ratio and using card for just oneoff transactions (may be for utility bills only).We can target them by providing less interest rate on Installment purchase transaction.
#    
# #### d. Group 3
#   - This group is performing best among all as cutomers are maintaining good credit score, doing both one off and instalment purchase,highest monthly average purchase and paying dues on time.
#       -- Giving rewards point will make them perform more purchases.
#        

# In[312]:


# Percentage of each cluster in the total customer base
s=cluster_df_4.groupby('Cluster_4').apply(lambda x: x['Cluster_4'].value_counts())
print (s,'\n')

per=pd.Series((s.values.astype('float')/ cluster_df_4.shape[0])*100,name='Percentage')
print ("Cluster -4 ",'\n')
print (pd.concat([pd.Series(s.values,name='Size'),per],axis=1),'\n')


# In[ ]:





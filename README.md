<H3>ENTER YOUR NAME : K S Ashwin Kumar</H3>
<H3>ENTER YOUR REGISTER NO :212224040034 </H3>
<H3>EX. NO.1</H3>
<H3>DATE : 12\05\26</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()

#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```

## OUTPUT:
<img width="1040" height="337" alt="image" src="https://github.com/user-attachments/assets/67c1d595-dce0-44d9-83a6-657a9b721b0a" />


<img width="425" height="652" alt="image" src="https://github.com/user-attachments/assets/cf638787-8d98-48dc-a1c2-848b1a02936f" />


<img width="451" height="603" alt="image" src="https://github.com/user-attachments/assets/03575fa7-cede-44c0-8ce6-34f406ecd60a" />


<img width="493" height="293" alt="image" src="https://github.com/user-attachments/assets/3ad1451d-d5d4-4337-834d-ebdadea20a17" />


<img width="613" height="468" alt="image" src="https://github.com/user-attachments/assets/0052ddcc-fec9-486a-8b45-078e5da80f15" />


<img width="1040" height="435" alt="image" src="https://github.com/user-attachments/assets/98a78729-b3f8-4e13-9a32-3b1b2cf37966" />


<img width="1013" height="582" alt="image" src="https://github.com/user-attachments/assets/ff5fc0ee-442d-41b3-8125-de806c2a70f8" />


<img width="813" height="335" alt="image" src="https://github.com/user-attachments/assets/9a77a7ff-bedd-4e98-b50d-779ba962781d" />


<img width="507" height="125" alt="image" src="https://github.com/user-attachments/assets/6ed95ef9-816a-4649-b875-ff6a8425c379" />


<img width="915" height="277" alt="image" src="https://github.com/user-attachments/assets/34bd4331-57c2-47bb-babe-ac7f5107a83f" />


<img width="420" height="113" alt="image" src="https://github.com/user-attachments/assets/1c4bc60a-4ce6-4e5a-97f4-ecf55d9dd1aa" />


<img width="923" height="336" alt="image" src="https://github.com/user-attachments/assets/bb86588e-6753-41f3-b99b-ef8d13629578" />





## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



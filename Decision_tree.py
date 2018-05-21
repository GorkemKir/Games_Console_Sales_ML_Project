import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
from sklearn import tree

#import graphviz

#Read Files
consoleSales = pd.read_csv('consoleSales.csv')
videoSales = pd.read_csv('Video_Game_Sales_as_of_Jan_2017.csv')

#Drop the NA values
videoSales = videoSales.dropna()

# Select all cases where PUBLISHER are nintende, microsoft, sony
videoSales = videoSales.loc[videoSales['Publisher'].isin(['Nintendo','Microsoft Game Studios'
                            ,'Sony Computer Entertainment'])]

# Select all cases where yeaR is greater than 2007
videoSales = videoSales [videoSales['Year_of_Release'] > 2007.0]
videoSales.drop(['Rating', 'Other_Sales', 'JP_Sales', 'EU_Sales', 'NA_Sales','Genre'], axis=1, inplace=True)

#Group by year and publisher and sum of Global Sales
videoSalesSum = videoSales.groupby(['Year_of_Release','Publisher'])['Global_Sales'].sum().reset_index()
#bar1=sns.barplot(x=videoSalesSum['Year_of_Release'], 
#            y=videoSalesSum['Global_Sales'],
#            hue=videoSalesSum['Publisher'], data=videoSalesSum,
#            hue_order=['Nintendo','Sony Computer Entertainment','Microsoft Game Studios'])
#plt.show(bar1)

#Console sales bar chart
#bar2=sns.barplot(x=consoleSales['Year'],
#            y=consoleSales['ConsoleUnitSold'], 
#            hue=consoleSales['Company'])
#plt.show(bar2)

#Group by two year and puslisher mean and sum
videoSalesMean = videoSales.groupby(['Year_of_Release','Publisher']).sum().reset_index()

#Merge the two file based on year and publisher
merged = pd.concat([videoSalesMean, consoleSales], axis=1)
merged.drop(['Company','Year'] ,axis=1, inplace=True)
merged.isnull().sum()

#Merge the global sales and console unit sales
#videoMean = videoSalesMean[['Year_of_Release','Publisher','Global_Sales']]
videoMean = videoSalesMean
videoMean = videoMean.rename(columns={'Year_of_Release':'Year', 'Publisher': 'Company' })
console_and_Global_Sale = pd.merge(consoleSales,videoMean,
                                   how='inner',
                                   on=['Year','Company'])
console_and_Global_Sale.drop(['Year','Company'], axis=1, inplace=True)

console_and_Global_Sale_2 = pd.merge(consoleSales,console_and_Global_Sale,
                                   how='inner',
                                   on=['ConsoleUnitSold'])

#console_and_Global_Sale_2.drop(['Company'], axis=1, inplace=True)

#ML Decision Tree

train, test = train_test_split(console_and_Global_Sale_2, test_size = 0.5); #Split the data
c = DecisionTreeClassifier(min_samples_split = 2, random_state = 0); #Now splits in every 100 so that overfitting doesnt occur

features = ["Global_Sales", "Critic_Score", "User_Score"];
trains = ["ConsoleUnitSold"];
X_Train = (np.round(train[features],5)).astype('int')
Y_Train = (np.round(train[trains],5)).astype('int')

X_Test = (np.round(test[features],5)).astype('int')
Y_Test = (np.round(test[trains],5)).astype('int')

decThree = c.fit(X_Train, Y_Train) #Produce Decision Tree

tree.export_graphviz(decThree, out_file = "assas.dot", feature_names = features)
 
y_pred = c.predict(X_Test)

IndexedTest = Y_Test.values
rowCount = Y_Test.size

error = 0

for x in range(0,rowCount):
    relEL = ((abs(y_pred[x] - IndexedTest[x][0])  / IndexedTest[x][0]  * 100))
    error = error + relEL
    
error = error / rowCount

totERR = (abs(Y_Test["ConsoleUnitSold"].sum() - y_pred.sum())  / Y_Test["ConsoleUnitSold"].sum()  * 100)

print("Average of Absulute Error: ", error, "%")
print("Total of Absulute Error: ", totERR, "%")

# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
print ("Accuracy : ", accuracy_score(Y_Test,y_pred)* 100, "%")


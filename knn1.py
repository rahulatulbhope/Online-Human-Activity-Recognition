import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import siganalysis as sa
import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import math
from scipy.fftpack import *
import scipy
from sklearn import metrics
#knn = neighbors.KNeighborsClassifier()





data_path = 'normaltrain1.csv'



file_data = np.loadtxt(data_path,delimiter=',')
data =[]
final_data= []
target_data = []


for i ,val in enumerate(file_data):
    data = [file_data[i][0],file_data[i][1],file_data[i][2],file_data[i][3],file_data[i][4],file_data[i][5]]
    final_data.append(data)
    target_data.append(file_data[i][6])
                       

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(final_data, target_data)

#print(neigh.predict([[1.003974259,0.177428076,1.518556173,0.738015228,0.780540945,100.2639644]]))

data_path1 = 'normaltest1.csv'



file_data1 = np.loadtxt(data_path1,delimiter=',')
data1 =[]
final_data1= []
target_data1 = []
count0 = 0
count1 = 0

for i ,val in enumerate(file_data1):
    data1 = [file_data1[i][0],file_data1[i][1],file_data1[i][2],file_data1[i][3],file_data1[i][4],file_data1[i][5]]
    final_data1.append(data1)
    target_data1.append(file_data1[i][6])
	
	
	
	
	

    
    

"""
for i in predict:
    
    if i == '[ 0.]':
        count0 =count0 + 1
    else:
        count1 = count1 + 1"""




from sklearn.metrics import classification_report
y_true, y_pred = target_data1, neigh.predict(final_data1)
print(classification_report(y_true, y_pred))
#########################Tuning Hyper Parameter with Accuracy Metrics[Test Data]#############################################
prediction = neigh.predict(final_data1)
    
print(metrics.accuracy_score(y_true,prediction))
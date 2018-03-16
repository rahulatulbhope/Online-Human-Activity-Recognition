# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn import metrics
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.



import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.utils import shuffle




data_path = 'normaltrain1.csv'



file_data = np.loadtxt(data_path,delimiter=',')
data =[]
final_data= []
target_data = []


for i ,val in enumerate(file_data):
    data = [file_data[i][0],file_data[i][1],file_data[i][2],file_data[i][3],file_data[i][4],file_data[i][5]]
    final_data.append(data)
    target_data.append(file_data[i][6])
	
	
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



from sklearn import preprocessing
from keras.utils.np_utils import to_categorical



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



rf = RandomForestClassifier(n_estimators=200,  n_jobs=4, min_samples_leaf=10)    
#train
rf.fit(final_data, target_data)

y_te_pred = rf.predict(final_data1)

acc = accuracy_score(target_data1, y_te_pred)
print("Random Forest Accuracy: %.5f" % (acc))














	

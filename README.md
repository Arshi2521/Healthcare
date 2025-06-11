# Healthcare
Hearth Dieases

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
​
df = pd.read_csv (r"C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv")
df
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	ASTHMA	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU
0	2	1	1	1	03-05-2020	3	1	65	2	0	...	0	0	1	0	0	0	0	0	3	0
1	2	1	2	1	03-06-2020	3	1	72	3	0	...	0	0	1	0	0	1	1	0	5	0
2	2	1	2	2	09-06-2020	1	2	55	3	1	...	0	0	0	0	0	0	0	0	3	1
3	2	1	1	1	12-06-2020	3	2	53	2	0	...	0	0	0	0	0	0	0	0	7	0
4	2	1	2	1	21-06-2020	3	2	68	3	1	...	0	0	1	0	0	0	0	0	3	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
199994	1	4	1	1	9999-99-99	3	2	27	2	0	...	0	0	0	0	0	0	0	0	6	0
199995	2	4	1	1	9999-99-99	3	2	42	2	0	...	0	0	0	0	0	0	0	0	6	0
199996	1	4	2	1	9999-99-99	3	2	57	3	0	...	0	0	0	0	0	0	0	1	6	0
199997	1	4	1	2	9999-99-99	2	2	45	2	0	...	0	0	0	0	0	0	0	0	6	1
199998	1	4	1	1	9999-99-99	3	2	54	2	0	...	0	0	0	0	0	1	0	0	6	0
199999 rows × 21 columns

df
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	ASTHMA	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU
0	2	1	1	1	03-05-2020	3	1	65	2	0	...	0	0	1	0	0	0	0	0	3	0
1	2	1	2	1	03-06-2020	3	1	72	3	0	...	0	0	1	0	0	1	1	0	5	0
2	2	1	2	2	09-06-2020	1	2	55	3	1	...	0	0	0	0	0	0	0	0	3	1
3	2	1	1	1	12-06-2020	3	2	53	2	0	...	0	0	0	0	0	0	0	0	7	0
4	2	1	2	1	21-06-2020	3	2	68	3	1	...	0	0	1	0	0	0	0	0	3	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
199994	1	4	1	1	9999-99-99	3	2	27	2	0	...	0	0	0	0	0	0	0	0	6	0
199995	2	4	1	1	9999-99-99	3	2	42	2	0	...	0	0	0	0	0	0	0	0	6	0
199996	1	4	2	1	9999-99-99	3	2	57	3	0	...	0	0	0	0	0	0	0	1	6	0
199997	1	4	1	2	9999-99-99	2	2	45	2	0	...	0	0	0	0	0	0	0	0	6	1
199998	1	4	1	1	9999-99-99	3	2	54	2	0	...	0	0	0	0	0	1	0	0	6	0
199999 rows × 21 columns

corona_dataset_csv = pd.read_csv(r"C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv")
corona_dataset_csv.head(10)
USMER	MEDICAL_UNIT	SEX	PATIENT_TYPE	DATE_DIED	INTUBED	PNEUMONIA	AGE	PREGNANT	DIABETES	...	ASTHMA	INMSUPR	HIPERTENSION	OTHER_DISEASE	CARDIOVASCULAR	OBESITY	RENAL_CHRONIC	TOBACCO	CLASIFFICATION_FINAL	ICU
0	2	1	1	1	03-05-2020	3	1	65	2	0	...	0	0	1	0	0	0	0	0	3	0
1	2	1	2	1	03-06-2020	3	1	72	3	0	...	0	0	1	0	0	1	1	0	5	0
2	2	1	2	2	09-06-2020	1	2	55	3	1	...	0	0	0	0	0	0	0	0	3	1
3	2	1	1	1	12-06-2020	3	2	53	2	0	...	0	0	0	0	0	0	0	0	7	0
4	2	1	2	1	21-06-2020	3	2	68	3	1	...	0	0	1	0	0	0	0	0	3	0
5	2	1	1	2	9999-99-99	2	1	40	2	0	...	0	0	0	0	0	0	0	0	3	1
6	2	1	1	1	9999-99-99	3	2	64	2	0	...	0	0	0	0	0	0	0	0	3	0
7	2	1	1	1	9999-99-99	3	1	64	2	1	...	0	1	1	0	0	0	1	0	3	0
8	2	1	1	2	9999-99-99	2	2	37	2	1	...	0	0	1	0	0	1	0	0	3	1
9	2	1	1	2	9999-99-99	2	2	25	2	0	...	0	0	0	0	0	0	0	0	3	1
10 rows × 21 columns

import pandas as pd
print(pd.__version__)
​
2.0.3
corona_dataset_csv.shape
(199999, 21)
corona_dataset_csv = pd.read_xlsx("C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.xlsx")
​
​
  Cell In[7], line 1
    corona_dataset_csv = pd.read_xlsx("C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.xlsx")
                                                                                                      ^
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape


corona_dataset_csv.size
pip install pandas numpy matplotlib seaborn scikit-learn
# Fill missing values
data.fillna(-1, inplace=True)
​
# Convert categorical features to numerical
data['sex'] = data['sex'].map({1: 'female', 2: 'male'})
data['patient type'] = data['patient type'].map({1: 'returned home', 2: 'hospitalization'})
​
import matplotlib.pyplot as plt
import seaborn as sns
​
# Example: distribution of ages
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
​
from sklearn.model_selection 
import train_test_split 
from sklearn.ensemble 
import RandomForestClassifier 
from sklearn.metrics 
import classification_report # Split the data X = data.drop('Classification_final', axis=1) y = data['Classification_final'] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train the model model = RandomForestClassifier() model.fit(X_train, y_train) 
# Evaluate the model y_pred = model.predict(X_test) print(classification_report(y_test,
import pandas as pd
​
# Try reading the file with a different encoding
data = pd.read_csv(r'C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv', encoding='latin1')
​
# Display the first few rows of the dataframe
print(data.head())
​
import pandas as pd
​
# Try reading the file with a different encoding
data = pd.read_csv(r"C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.xlsx", encoding='latin1')
​
# Display the first few rows of the dataframe
print(data.head())
​
import pandas as pd
import csv
​
# Load the data with the quoting parameter
data = pd.read_csv(r'C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv', encoding='latin1', quoting=csv.QUOTE_NONE)
​
# Display the first few rows of the dataframe
print(data.head())
​
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
​
# Load the data
data = pd.read_csv(r'C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv', encoding='latin1')
​
# Replace '9999-99-99' with NaN
data['DATE_DIED'] = data['DATE_DIED'].replace('9999-99-99', np.nan)
​
# Convert DATE_DIED to datetime
data['DATE_DIED'] = pd.to_datetime(data['DATE_DIED'], errors='coerce')
​
# Drop the DATE_DIED column if it's not needed for the model
data = data.drop(columns=['DATE_DIED'])
​
# Convert categorical features to numerical
data['SEX'] = data['SEX'].map({'female': 0, 'male': 1})  # 0 for female, 1 for male
data['PATIENT_TYPE'] = data['PATIENT_TYPE'].map({'returned home': 0, 'hospitalization': 1})  # 0 for returned home, 1 for hospitalization
​
# Fill missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
​
# Split the data into features and target
X = data_imputed.drop('CLASIFFICATION_FINAL', axis=1)
y = data_imputed['CLASIFFICATION_FINAL']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Initialize the model
model = RandomForestClassifier()
​
# Fit the model
model.fit(X_train, y_train)
​
# Make predictions
y_pred = model.predict(X_test)
​
# Evaluate the model
print(classification_report(y_test, y_pred))
​
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
​
# Load the data
data = pd.read_csv(r'C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv', encoding='latin1')
 
    
print(data.head())
​
# Fill missing values
data.fillna(-1, inplace=True)
​
# Convert categorical features to numerical
data['SEX'] = data['SEX'].map({1: 0, 2: 1})  # 0 for female, 1 for male
data['PATIENT_TYPE'] = data['PATIENT_TYPE'].map({1: 0, 2: 1})  # 0 for returned home, 1 for hospitalization
​
# Split the data into features and target
X = data.drop('CLASIFFICATION_FINAL', axis=1)
y = data['CLASIFFICATION_FINAL']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Initialize the model
model = RandomForestClassifier()
​
# Fit the model
model.fit(X_train, y_train)
​
# Make predictions
y_pred = model.predict(X_test)
​
# Evaluate the model
print(classification_report(y_test, y_pred))
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
​
# Initialize the model
model = HistGradientBoostingClassifier()
​
# Fit the model
model.fit(X_train, y_train)
​
# Make predictions
y_pred = model.predict(X_test)
​
# Evaluate the model
print(classification_report(y_test, y_pred))
​
# Drop rows with missing values
data = data.dropna()
​
# Split the data into features and target
X = data.drop('CLASIFFICATION_FINAL', axis=1)
y = data['CLASIFFICATION_FINAL']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Initialize the model
model = RandomForestClassifier()
​
# Fit the model
model.fit(X_train, y_train)
​
# Make predictions
y_pred = model.predict(X_test)
​
# Evaluate the model
print(classification_report(y_test, y_pred))
​
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
​
# Load the data
data = pd.read_csv(r'C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv', encoding='latin1')
​
# Display the first few rows of the dataframe
print(data.head())
​
# Replace '9999-99-99' with NaN
data['DATE_DIED'] = data['DATE_DIED'].replace('9999-99-99', np.nan)
​
# Convert DATE_DIED to datetime
data['DATE_DIED'] = pd.to_datetime(data['DATE_DIED'], errors='coerce')
​
# Drop the DATE_DIED column if it's not needed for the model
data = data.drop(columns=['DATE_DIED'])
​
# Fill missing values
data.fillna(-1, inplace=True)
​
# Convert categorical features to numerical
data['SEX'] = data['SEX'].map({'female': 0, 'male': 1})  # 0 for female, 1 for male
data['PATIENT_TYPE'] = data['PATIENT_TYPE'].map({'returned home': 0, 'hospitalization': 1})  # 0 for returned home, 1 for hospitalization
​
# Split the data into features and target
X = data.drop('CLASIFFICATION_FINAL', axis=1)
y = data['CLASIFFICATION_FINAL']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
# Initialize the model
model = RandomForestClassifier()
​
# Fit the model
model.fit(X_train, y_train)
​
# Make predictions
y_pred = model.predict(X_test)
​
# Evaluate the model
print(classification_report(y_test, y_pred))
​
import matplotlib.pyplot as plt
import seaborn as sns
​
# Visualize the distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True)
plt.title('Age Distribution of COVID-19 Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
​
# Visualize the proportion of different classifications
plt.figure(figsize=(10, 6))
sns.countplot(data['Classification_final'])
plt.title('COVID-19 Disease Classification')
plt.xlabel('Classification')
plt.ylabel('Count')
plt.show()
​
import matplotlib.pyplot as plt
​
# Plotting example
plt.figure(figsize=(10, 6))
plt.plot(data['AGE'], data['PNEUMONIA'])
plt.title('Age vs Pneumonia')
plt.xlabel('Age')
plt.ylabel('Pneumonia')
plt.show()
​
import matplotlib.pyplot as plt
​
# Plotting example with the correct column name
plt.figure(figsize=(10, 6))
plt.plot(data['AGE'], data['PNEUMONIA'])
plt.title('Age vs Pneumonia')
plt.xlabel('Age')
plt.ylabel('Pneumonia')
plt.show()
​
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
​
​
data = pd.read_csv(r'C:\Users\arshi\OneDrive\Desktop\Health care\Covid_Dataset.csv', encoding='latin1')
​
​
print(data.head())
​
​
data.fillna(-1, inplace=True)
​
​
data['SEX'] = data['SEX'].map({1: 'female', 2: 'male'})
data['PATIENT_TYPE'] = data['PATIENT_TYPE'].map({1: 'returned home', 2: 'hospitalization'})
​
​
X = data.drop('CLASIFFICATION_FINAL', axis=1)
y = data['CLASIFFICATION_FINAL']
​
​
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​
​
model = RandomForestClassifier()
​
​
model.fit(X_train, y_train)
​
​
y_pred = model.predict(X_test)
​
​
print(classification_report(y_test, y_pred))
​

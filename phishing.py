# importing data manipulation language
import pandas as pd
import numpy as np

# getting data 
df=pd.read_csv('https://raw.githubusercontent.com/Yash-Pandey007/Phishing_ML/refs/heads/main/dataset_phishing.csv')

# identifing constant Features
un_data={col:len(df[col].unique()) for col in df.columns}

# Rmoving constant features
for col, val in un_data.items():
  if val==1:
    df.drop(col, axis=1, inplace=True)

#library for Feature Exratction
from urllib.parse import urlparse, parse_qs
import socket

# Function to extract features
def extract_url_features(url):
    parsed = urlparse(url)
    # Feature extraction
    features = {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'url_length': len(url),
        'num_path_segments': len([seg for seg in parsed.path.split('/') if seg]),
        'num_query_params': len(parse_qs(parsed.query)),
    }
    # Check if domain is an IP address
    try:
        socket.inet_aton(parsed.netloc)
        features['is_ip'] = 1
    except Exception:
        features['is_ip'] = 0
    return features

# Apply extraction function
features_df = df['url'].apply(extract_url_features).apply(pd.Series)
df = pd.concat([df, features_df], axis=1)

corr_matrix = df.select_dtypes(include=np.number).corr()
threshold = 0.75  # Adjust the threshold as needed
correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            correlated_features.add(colname)

# Remove correlated features (you can choose to keep one of the correlated features)
df_selected = df.drop(columns=correlated_features)

# features slection based on variances
numerical_features = df_selected.select_dtypes(include=np.number).columns
variances = df_selected[numerical_features].var()
variance_threshold = variances.quantile(0.50)
selected_features = variances[variances > variance_threshold].index
df_selected = df[selected_features]
df_selected.head()

# creating the x and y features
x=pd.merge(df_selected, features_df, left_index=True, right_index=True)
y=df['status']

# Encoding the String Features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['scheme'] = le.fit_transform(x['scheme'])
x['netloc'] = le.fit_transform(x['netloc'])
x['path'] = le.fit_transform(x['path'])

# Transforming the using the yeo-johnson
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
x_yeo_johnson = pt.fit_transform(x)
x = pd.DataFrame(x_yeo_johnson, columns = x.columns)

# spliting the data in testing and training 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#model training
def model_evaluation(model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test):
  from sklearn.metrics import confusion_matrix, classification_report,f1_score
  model.fit(x_train, y_train)
  y_pred_train = model.predict(x_train)
  y_pred_test = model.predict(x_test)
  print("Training Metrics:")
  print(classification_report(y_train, y_pred_train))
  print(confusion_matrix(y_train, y_pred_train))
  print(f1_score(y_train, y_pred_train,pos_label='phishing'))
  print("Testing Metrics:")
  print(classification_report(y_test, y_pred_test))
  print(confusion_matrix(y_test, y_pred_test))
  print(f1_score(y_test, y_pred_test,pos_label='phishing'))


from sklearn.svm import SVC
svc = SVC(C=0.13066739238053282,gamma=0.05399484409787434)
model_evaluation(svc)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import numpy as np

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None 

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def dropNA(self):
        self.data.dropna(inplace=True)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def dropCol(self, cols):
        self.input_data = self.input_data.drop(cols, axis=1)
             
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def gender_encoding(self):
        map = {"Gender": {"Male": 1, "Female": 0}}
        self.x_train = self.x_train.replace(map)
        self.x_test = self.x_test.replace(map)
      
    def onehot(self, col):
        encoder = OneHotEncoder(handle_unknown="ignore")
        subs_enc_train = self.x_train[[col]]
        subs_enc_test = self.x_test[[col]]

        train_encoded = pd.DataFrame(encoder.fit_transform(subs_enc_train).toarray(), columns=encoder.get_feature_names_out())
        test_data = pd.DataFrame(encoder.transform(subs_enc_test).toarray(), columns=encoder.get_feature_names_out())
        self.x_train = self.x_train.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)
        self.x_train = pd.concat([self.x_train, train_encoded], axis=1)
        self.x_test = pd.concat([self.x_test, test_data], axis=1)
        
        self.x_train.drop(col, axis=1, inplace=True)
        self.x_test.drop(col, axis=1, inplace=True)
    
    def createMean(self,col):
        return np.mean(self.x_train[col])
    
    def replace_num(self, col, num):
        self.x_train[col] = self.x_train[col].replace(0,num)
        self.x_test[col] = self.x_test[col].replace(0,num)
    
    def scaler(self, columns):
        scaler = MinMaxScaler()
        self.x_train[columns] = scaler.fit_transform(self.x_train[columns])
        self.x_test[columns] = scaler.transform(self.x_test[columns])

    def createModel(self,learning_rate=0.1,maxdepth=6,est=100):
         self.model = XGBClassifier(learning_rate=learning_rate,max_depth=maxdepth,n_estimators=est)
  
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0','1']))



file_path = 'data_D.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn')

# Drop null values
data_handler.dropNA()

input_df = data_handler.input_df
output_df = data_handler.output_df

# Drop columns
model_handler = ModelHandler(input_df, output_df)
model_handler.dropCol(['Unnamed: 0','id','CustomerId','Surname'])

# Data Splitting
model_handler.split_data()

# Encoding
model_handler.gender_encoding()
model_handler.onehot('Geography')

# Outlier handling
balance_mean = model_handler.createMean('Balance')
model_handler.replace_num('Balance', balance_mean)

# Scaling
model_handler.scaler(['CreditScore', 'Balance', 'EstimatedSalary'])

# Modelling
model_handler.createModel()
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()





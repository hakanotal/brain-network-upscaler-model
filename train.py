import csv
import json
import numpy as np
import random as r
import pandas as pd
import matplotlib.pyplot as plt 

from sklearnex import patch_sklearn
patch_sklearn()

from time import time
from sklearn import feature_selection 
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor,RegressorChain
from sklearn.linear_model import LinearRegression,Ridge,Lasso,MultiTaskLasso,ElasticNet,MultiTaskElasticNet,RidgeCV
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold,validation_curve
from sklearn.feature_selection import VarianceThreshold

r.seed(1)


class Feature_select:
    '''
    Feature selection 
    '''
    def __init__(self,variance_th = True, effective_w =  True):
        self.params_feature_selection = {
            "Variance_threshold" : 0.5 * 10e-3,
            "Threshold of effectiwe weights" : 0.2,
            "Thresold for correlation": 0.79,
            }

        self.variance_th = variance_th
        self.effective_w = effective_w
        self.vt = VarianceThreshold(threshold=self.params_feature_selection["Variance_threshold"])
        self.indexes_ew = None # keep indexes for 
        self.pca = PCA() # maximum features that can have 

    
    def select_effective_weights(self,train_x,train_y,th):
        '''
        Simply training a simple model and check effects of features than eliminate ineffective ones
        '''
        ranking_f = np.zeros((train_x.shape[1])) # voting for all features 
        regressors = MultiOutputRegressor(estimator = DecisionTreeRegressor(max_features = "sqrt"),n_jobs = -1)
        regressors.fit(train_x,train_y)     

        for estimator in regressors.estimators_:
            ranking_f += estimator.feature_importances_

        ranking_f = ranking_f/max(ranking_f)
        self.indexes_ew = np.nonzero(ranking_f > th)
        return train_x[:,self.indexes_ew].reshape((train_x.shape[0],-1))
    
    def eliminate_high_corr_f(self,train_x,th):
        """ Eliminate highly correlated featueres so it will give better results in PCA

        Args:
            train_x (numpy.ndarray): Data set to be filtered 
            th (float): Threshold for filtering 

        Returns:
            numpy.ndarray: Filtered set
        """
        cov_mat = np.corrcoef(train_x,rowvar=False)
        cov_mat_upper = np.triu(cov_mat)
        np.fill_diagonal(cov_mat_upper,0)
        '''
        X and Y values are highly correlated features
        We can eliminate only one of them {Just X or Just X not bot of them}
        '''
        self.x,self.y = np.nonzero(cov_mat_upper > th)

        return np.delete(train_x,self.x,axis=1)
         
    def fit(self,data_x,data_y):
        if(self.variance_th):
            data_x = self.vt.fit_transform(data_x)
            print(" ")
            print("Shape of x_train after variance thresholding : {}".format(data_x.shape))
        data_x = self.eliminate_high_corr_f(data_x,self.params_feature_selection["Thresold for correlation"])
        print("Shape of x_train after eliminate high correlation_features : {}".format(data_x.shape))
        if(self.effective_w):
            data_x = self.select_effective_weights(data_x,data_y,self.params_feature_selection["Threshold of effectiwe weights"])
            print(" ")
            print("Shape of x_train after select effective weights : {}".format(data_x.shape))

        print(data_x.shape)

        data_x = self.pca.fit_transform(data_x)

        return data_x
    
    def transform(self,data_x):
        if(self.variance_th):
            data_x = self.vt.transform(data_x)
            print(data_x.shape)
        data_x = np.delete(data_x,self.x,axis=1)
        if(self.effective_w):
            data_x = data_x[:,self.indexes_ew].reshape((data_x.shape[0],-1))
            print(data_x.shape)
        data_x = self.pca.transform(data_x)
        return data_x



class Pipeline:
    '''
    ML pipeline for predicting high resolution graphs from low resolution graph
    '''
    def __init__(self):

        '''
        Read csv files
        '''
        self.data_lr = pd.read_csv("data/train_LR.csv")
        self.data_hr = pd.read_csv("data/train_HR.csv")

        self.kaggle_lr = pd.read_csv("data/test_LR.csv")

        print("********* Data is loaded *********\n")
        '''
        Make them numpy array
        '''
        self.data_x = np.array(self.data_lr) # inputun 
        self.data_y = np.array(self.data_hr) #Ground truth 

        self.test_x = np.array(self.kaggle_lr)

        self.fs = Feature_select()

        self.name_of_algorithm = "MultiTaskElasticNet"
        self.algorithm = MultiTaskElasticNet
        self.losses_mse_test = np.zeros(5)
        self.params_model = {
            "alpha":1.2,
            "l1_ratio":1,
        }
        self.params_features = {
            "Pca" : True , # Always True
            "Variance Th" : True,
            "effective_w" :  False # take some time 
        }
   
    def kaggle_output(self):

        '''
        Training model with the whole input 
        Model see not 4/5 of the input instead it take information from all inputs 
        '''
        print("\n-----Training for kaggle input starting-----\n")
        r1 = ElasticNet()
        r2 = LinearSVR()
        r3 = Ridge()
        reg = VotingRegressor([('r1', r1), ('r2', r2), ('r3', r3)])
        final_model = MultiOutputRegressor(reg, n_jobs=-1)

        fs = Feature_select(self.params_features["Variance Th"],self.params_features["effective_w"])
        train_x = fs.fit(self.data_x,self.data_y)
        train_y = self.data_y
        print("Shape of train_x :{} ".format(train_x.shape))
        print("Shape of train_y :{} ".format(train_y.shape))

        final_model.fit(train_x,train_y)

        '''
        Predicting test inputs
        '''
        if(self.params_features["Feature extraction"]):
            self.test_x = self.feature_extraction(self.test_x)
        
        print("\n-----Predicting Test Input-----\n")
        print("Shape of test_x :{} ".format(self.test_x.shape))
        self.test_x = fs.transform(self.test_x)
        print("Shape of test_x before predicting :{} ".format(self.test_x.shape))
        results = final_model.predict(self.test_x)
        print("Shape of results:{} ".format(results.shape))
        results = self.fix_outlier(results)
        results = results.flatten()

        filename = "output_{}.csv".format(self.name_of_algorithm)
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "predicted"])
            for i in range(results.size):
                writer.writerow([i, results[i]])

    def save_results(self):
        with open('results.txt', 'a') as file:
            file.write("Algorithm : {} \n".format(self.name_of_algorithm))
            file.write(json.dumps(self.params_model)) # use `json.loads` to do the reverse
            file.write("\n")
            file.write(json.dumps(self.params_features)) # use `json.loads` to do the reverse
            file.write("\n")
            file.write(json.dumps(self.fs.params_feature_selection)) # use `json.loads` to do the reverse
            file.write("\nMSE_train scores: \n")
            for i in self.losses_mse_train:
                file.write(str(i) + " ")
            file.write("\n")
            file.write("Avg of nMSE_train scores:{}\n".format(np.mean(self.losses_mse_train)))
            file.write("\nMSE_test scores: \n")
            for i in self.losses_mse_test:
                file.write(str(i) + " ")
            file.write("\n")
            file.write("Avg of MSE_test scores:{}\n".format(np.mean(self.losses_mse_test)))
            file.write("\MAD_test scores: \n")
            for i in self.mad_test:
                file.write(str(i) + " ")
            file.write("\n")
            file.write("Avg of MAD_test scores:{}\n".format(np.mean(self.mad_test)))

            file.write("\n\n")

    def fix_outlier(self,data_x):
        '''
        All distribution is in interval (0,1) 
        '''
        data_x[data_x < 0] = 0.002 # leaky clamping
        data_x[data_x > 1] = 0.98  # leaky clamping

        return data_x
    
    def mad(self,predicted,ground_truth):
        '''
        Mean absolute Distance Error 
        '''
        flat_pred = predicted.flatten()
        flat_gt = ground_truth.flatten()
        sum = 0
        n = predicted.shape[0]
        for i in range(n):
            for j in range(n):
                sum += abs(flat_pred[i] - flat_gt[j])
        return sum/(n**2)

    def train(self):
        k = 5
        
        '''
        Change it bcs of the memory issue insted of keeping all models just keep min and current model
        '''
        r1 = ElasticNet()
        r2 = LinearSVR()
        r3 = Ridge()
        reg = VotingRegressor([('r1', r1), ('r2', r2), ('r3', r3)])
        # reg = StackingRegressor(estimators=[('r1', r1), ('r2', r2), ('r3', r3)], final_estimator=LinearRegression())
        # self.regressors = [reg for i in range(k)] # for multitask Elastic net
        # self.regressors = [MultiOutputRegressor(reg) for i in range(k)] # for multitask Elastic net

        self.mad_test = np.zeros(k)
        self.losses_mse_train = np.zeros(k)
        self.pearson_test = []

        self.losses_mse_test = np.zeros(k)
        self.mad_train = np.zeros(k)
        self.pearson_train = []

        # fs_array = [Feature_select(self.params_features["Variance Th"],self.params_features["effective_w"]) for i in range(k)]


        attempt_num = 0
        kf = KFold(n_splits=k)
        for train_index,test_index in kf.split(self.data_x,self.data_y):
            '''
            Some algorithms have too much coefs and all 5 model won't fit ram 
            so instead of keeping fitted model in ram save it to hard disk 
            then you can load it back
            '''
            regressor = MultiOutputRegressor(reg, n_jobs=-1)
            fs = Feature_select(self.params_features["Variance Th"],self.params_features["effective_w"])
            print("\n**** Fold number {} is started ****\n".format(attempt_num+1))

            x_train = self.data_x[train_index]
            y_train = self.data_y[train_index]

            x_test = self.data_x[test_index]
            y_test = self.data_y[test_index]

            print("----Feature Selection Started----")
            print("Shape of x_train before feature selection : {}".format(x_train.shape))
            print("Shape of x_test before feature selection : {}\n".format(x_test.shape))
            reduced_x_train =  fs.fit(x_train,y_train)
            reduced_x_test =  fs.transform(x_test)
            print("----Feature Selection Ended----")
            print("Shape of x_train after feature selection : {}".format(reduced_x_train.shape))
            print("Shape of x_test after feature selection : {}".format(reduced_x_test.shape))

            print("\n***********************************************")
            print("************** Training starting **************")
            print("***********************************************\n")

            print("Shape of x_train: {}".format(reduced_x_train.shape))
            print("Shape of y_train: {}\n".format(y_train.shape))

            regressor.fit(reduced_x_train,y_train)

            train_pred = regressor.predict(reduced_x_train)
            train_pred = self.fix_outlier(train_pred)
            self.losses_mse_train[attempt_num] = mean_squared_error(y_train,train_pred)
            self.mad_train[attempt_num] = self.mad(train_pred,y_train)

            test_pred = regressor.predict(reduced_x_test)
            test_pred = self.fix_outlier(test_pred)
            self.losses_mse_test[attempt_num] = mean_squared_error(y_test,test_pred)
            self.mad_test[attempt_num] = self.mad(test_pred,y_test)

            print("Loss mse is :", str(self.losses_mse_test[attempt_num]))
            attempt_num+=1

        print("Avg error is {}".format(np.mean(self.losses_mse_test)))

        self.save_results() # write to txt


p = Pipeline()

t1 = time()
# p.train()
p.kaggle_output()
t2 = time()
print(f'Training completed in {(t2-t1):.4f}s')
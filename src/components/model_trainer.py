import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,):
        try:
            logging.info("Split training and test input data")
            
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            models = {
                "Linear Regression"        : LinearRegression(),
                "K-Neighbors Regressor"   : KNeighborsRegressor (),
                "Desision Tree Regressor"  : DecisionTreeRegressor (),
                "Random Forest Regressor"  : RandomForestRegressor(),
                "AdaBoost Regressor"       : AdaBoostRegressor(),
                "Gradient Boost Regressor" : GradientBoostingRegressor(),
                "Support Vector Regressor" : SVR(),
                "CatBoosting Regressor"    : CatBoostRegressor(),
                "XGBRegressor"             : XGBRegressor()
            }

            params ={
                "Linear Regression": {},
                "K-Neighbors Regressor":{'n_neighbors':range(1,11), 
                                            'weights' : ['uniform','distance'], 
                                            'p': [1, 2]
                                        },
                "Desision Tree Regressor":{'splitter': ['best','random'], 
                                           'max_depth': [None,300,500,700], 
                                           'min_samples_split' : [2,5,8,10], 
                                           'min_samples_leaf' : [2,5,8,10,20], 
                                           'max_leaf_nodes': [4,8, None]
                                          },
                "Random Forest Regressor":{'n_estimators' : [50,100,200],
                                           'max_depth' : [50,100,200],
                                           'min_samples_split'  : [2,8,10],
                                           'min_samples_leaf' : [2,8,10], 
                                           'max_leaf_nodes': [4,8, None]
                                          },
                "AdaBoost Regressor": {'n_estimators': [8,16,32,64,128,256],
                                       'learning_rate':[.1,.01,0.5,1]
                                      },
                "Gradient Boost Regressor":{#'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                                            'learning_rate':[.1,.01,.05,.001],
                                            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                                            # 'criterion':['squared_error', 'friedman_mse'],
                                            # 'max_features':['auto','sqrt','log2'],
                                            'n_estimators': [8,16,32,64,128,256]
                                           },
                "Support Vector Regressor" : {},
                "CatBoosting Regressor"    : {'depth': [6,8,10],
                                              'learning_rate': [0.01, 0.05, 0.1],
                                              'iterations': [30, 50, 100]
                                             },
                "XGBRegressor": {'learning_rate':[.1,.01,.05,.001],
                                 'n_estimators': [8,16,32,64,128,256]
                                }
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,
                                               X_test=X_test,y_test=y_test,
                                               models=models,params=params)

            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the best model with the best score from dict
            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path ,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
        
    



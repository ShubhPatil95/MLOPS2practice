# load the train and test
# train algo
# save the metrices, params

from get_data import read_params
import pandas as pd
from sklearn.linear_model import ElasticNet
import json
import os
import joblib
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config=read_params(config_path)
    train_data_path=config["split_data"]["train_path"]
    test_data_path=config["split_data"]["test_path"]
    
    train=pd.read_csv(train_data_path)
    test=pd.read_csv(test_data_path)
    
    target=config["base"]["target_col"]
    
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    random_state=config["base"]["random_state"]

    trainy=train[target]
    testy=test[target]
    
    trainx=train.drop(target,axis=1)
    testx=test.drop(target,axis=1)
    
    lr=ElasticNet(alpha=alpha,
                  l1_ratio=l1_ratio,
                  random_state=random_state)
    lr.fit(trainx,trainy)
    
    predicted_qualities=lr.predict(testx)
    
    (rmse, mae, r2) = eval_metrics(testy, predicted_qualities)
    
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#####################################################
    scores_file=config["reports"]["scores"]
    params_file=config["reports"]["params"]
    
    with open(scores_file,"w") as score:
        scores={
            "rmse":rmse,
            "mase": mae,
            "r2": r2
            }
        json.dump(scores,score,indent=4)
        
    with open(params_file, "w") as param:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, param, indent=4)
#####################################################

    model_dir=config["model_dir"]
    
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(lr, model_path)
    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/home/shubham/MLOSP2practice/simple_app/params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
from _datetime import datetime
import json
from logging import getLogger, config
import sys

from boatrace.common.BoatEnum import DelimiterType
from boatrace.util.PropertyUtil import PropertyUtil
import pandas as pd
import math
from sklearn.metrics._ranking import ndcg_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import lightgbm as lgb

#
# model生成するクラス
#
class BoatLGBMRankerTest:
    def __init__(self):
        self._logger = getLogger('server')
        
    # @param param_list_str ex)  boosting_type=gbdt,learning_rate=0.1...
    # @param csv_filepath ex)  D:/Dev/experiment/expr10/arff/9997_rank1.csv
    # @param model_filepath ex)  D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model
    # @param feature_name_list_str ex)  nw1,nw2,nw3,nw4,nw5,nw6,class
    # @param feature_type_list_str ex)  float,float,float,float,float,float,category
    def execute(self, param_list_str, csv_filepath, model_filepath, feature_name_list_str, feature_type_list_str):
        
        #feature명 리스트 취득
        feature_name_list = feature_name_list_str.split(DelimiterType.DELIM_COMMA.value)
        
        #feature type 리스트 취득
        feature_type_list = feature_type_list_str.split(DelimiterType.DELIM_COMMA.value)
        
        #csv data type 設定
        #data_type = {'nw1': 'float', 'nw2': 'float', 'nw3': 'float', 'nw4': 'float', 'nw5': 'float', 'nw6': 'float', 'class': 'category'}
        feature_num = len(feature_name_list)
        data_type_dict = {}
        cat_features = []
        for i in range(feature_num):
            data_type_dict[feature_name_list[i]] = feature_type_list[i]
            if feature_type_list[i] == 'category'  and feature_name_list[i] != 'class':
                cat_features.append(feature_name_list[i])
        
        # csv data 取得
        df = pd.read_csv(csv_filepath, names=feature_name_list, dtype=data_type_dict, engine='python')
        df = df.sort_values([ 'raceid', 'waku'])
        
        keys = df['raceid'].unique()
        date_split_val =  keys[math.trunc( len(keys) * 0.6)]
        date_split_test =  keys[math.trunc( len(keys) * 0.8)]
        
        # date_split_val = 202006020000
        # date_split_test = 202106012400
        
        #test_size = 0.2
                
        df_train = df[df['raceid'] <= date_split_val]
        #df_val = df[ (df['raceid'] > date_split_val)]
        df_val = df[ (df['raceid'] > date_split_val) & (df['raceid'] < date_split_test) ]
        df_test = df[df['raceid'] >= date_split_test]
        print( "df_train length="  + str(len(df_train)))
        print( "df_val length="  + str(len(df_val)))
        print( "df_val length="  + str(len(df_test)))
        
        X_train = df_train.drop(['class'], axis=1)
        X_val = df_val.drop(['class'], axis=1)
        X_test = df_test.drop(['class'], axis=1)
        
        y_train = df_train['class']
        y_val = df_val['class']
        y_test = df_test['class']
        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
    
        #モデル파라미터 설정
        #model_param_list = param_list_str.split(DelimiterType.DELIM_COMMA.value)
        #model_param_dict = {}
        #for param in model_param_list:
        #  key, value = param.split(DelimiterType.DELIM_EQUAL.value)
        #    model_param_dict[key] = value

        # 모델 생성
        #model = cab.CatBoostRanker(**model_param_dict)

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights))
        print(class_weights)
            
        param = {'learning_rate'  : 0.01}
        param['class_weight'] = class_weights

        #param = {'iterations': 100}   
        # 모델 생성
        model = lgb.LGBMRanker(**param)
        
        train_group = X_train['raceid'].value_counts()
        train_group = train_group.sort_index()
        
        eval_group = X_val['raceid'].value_counts()
        eval_group = eval_group.sort_index()
        
        # model.fit(X_train, y_train)
        model.fit(X_train, y_train, group=train_group,  eval_set=[(X_val, y_val)],  eval_group=[list(eval_group)])
        print(model.get_params())

        feature = list(df.drop(columns=['class']).columns)
        importances = np.array(model.feature_importances_)
        
        df = pd.DataFrame({'feature':feature, 'importance':importances})
        df = df.sort_values('importance', ascending=False)
        for row in df.iterrows():
            print(row);

        y_expected  = y_test
        y_predicted = model.predict(X_test)
        
        #print(model.get_test_eval())
        #print(model.get_evals_result())
        print(ndcg_score([y_expected], [y_predicted]))
        
        probabilities = self._ranking_scores_to_probabilities(y_predicted)
        #print(X_test)
        for i in range(24) :
            print (i+1, round(y_predicted[i], 3), round(probabilities[i], 3))

    def _ranking_scores_to_probabilities(self, scores) -> list[float]:
        """
        ランキングスコアを確率に変換する関数
    
        :param scores: ランキングスコアのリスト
        :return: 確率に変換されたスコアのリスト
        """
        # スコアを指数関数で変換
        exp_scores = np.exp(scores)
        
        # 正規化して確率に変換
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
        
def logSetup():
    """
    loggin環境を定義する
    """ 
    prop = PropertyUtil.getInstance()
    with open(prop.getProperty('file_python_log_config'), 'r', encoding='utf-8') as f:
        log_config = json.load(f)
        
    # ファイル名をタイムスタンプで作成
    log_file_name = prop.getProperty('file_python_log')
    log_config["handlers"]["fileHandler"]["filename"] = log_file_name.format(datetime.utcnow().strftime("%Y%m%d"))

    config.dictConfig(log_config)
    
def main(argv):
    # argv: {params} {csv_file_path} {model_file_path} {feature_name_list} {feature_type_list}
    # ex: boosting_type=gbdt,learning_rate=0.1 D:/Dev/experiment/expr10/arff/9997_rank1.csv D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model nw1,nw2,nw3,nw4,nw5,nw6,class float,float,float,float,float,float,category
    
    #param len check
    if len(argv) < 6:
        print('Usage: python xxxModelGenerator.py {params} {csv_file_path} {model_file_path} {feature_name_list} {feature_type_list}')
        return -1
    
    prop = PropertyUtil.getInstance()
    prop.addFile('C:/Dev/github/pod_boatrace/properties/expr10/expr10.properties')
    
    # loggin set up
    logSetup()

    tester = BoatLGBMRankerTest()
    tester.execute(argv[1], argv[2], argv[3], argv[4], argv[5])
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


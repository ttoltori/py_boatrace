from _datetime import datetime
import json
from logging import getLogger, config
import pickle
import sys

from sklearn import metrics
from sklearn.model_selection._split import train_test_split

from boatrace.common.BoatEnum import DelimiterType
from boatrace.util.PropertyUtil import PropertyUtil
import lightgbm as lgb
import pandas as pd


#
# model生成するクラス
#
class BoatLGBMRegressorTest:
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
        for i in range(feature_num):
            data_type_dict[feature_name_list[i]] = feature_type_list[i]
        
        # csv data 取得
        df = pd.read_csv(csv_filepath, names=feature_name_list, dtype=data_type_dict, engine='python')
        
        # train data 取得
        x = df[feature_name_list[0:feature_num-1]]
        y = df[feature_name_list[feature_num-1]]
        
        #モデル파라미터 설정
        model_param_list = param_list_str.split(DelimiterType.DELIM_COMMA.value)
        model_param_dict = {}
        for param in model_param_list:
            key, value = param.split(DelimiterType.DELIM_EQUAL.value)
            model_param_dict[key] = value
            
        # 모델 생성
        model = lgb.LGBMRegressor(**model_param_dict)

        # モデル学習
        model.fit(x, y)
        
        # # モデル保存
        pickle.dump(model, open(model_filepath, 'wb'))
        #
        self._logger.info('model created:' + model_filepath)
        
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

    trainer = BoatLGBMRegressorTest()
    trainer.execute(argv[1], argv[2], argv[3], argv[4], argv[5])
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


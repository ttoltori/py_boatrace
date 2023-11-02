from _datetime import datetime
import json
from logging import getLogger, config
import pickle
import sys

from boatrace.util.PropertyUtil import PropertyUtil
import lightgbm as lgb
import pandas as pd
from tmp import LGBMTrainer

#
# model生成するクラス
#
class LGBMTraLGBMClassifierTrainer():
    DELIM_COMMA = ','
    DELIM_EQUAL = '='
    
    # @param param_list_str ex)  boosting_type=gbdt,learning_rate=0.1...
    # @param csv_filepath ex)  D:/Dev/experiment/expr10/arff/9997_rank1.csv
    # @param model_filepath ex)  D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model
    # @param feature_name_list_str ex)  nw1,nw2,nw3,nw4,nw5,nw6,class
    # @param feature_type_list_str ex)  float,float,float,float,float,float,category
    def execute(self, param_list_str, csv_filepath, model_filepath, feature_name_list_str, feature_type_list_str):
        
        #feature명 리스트 취득
        feature_name_list = feature_name_list_str.split(self.DELIM_COMMA)
        
        #feature type 리스트 취득
        feature_type_list = feature_type_list_str.split(self.DELIM_COMMA)
        
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
        model_param_list = param_list_str.split(self.DELIM_COMMA)
        model_param_dict = {}
        for param in model_param_list:
            key, value = param.split(self.DELIM_EQUAL)
            model_param_dict[key] = value
            
        # 모델 생성
        model = lgb.LGBMClassifier(**model_param_dict)

        # モデル学習
        model.fit(x, y)
        
        # x_test = [[6.32,4.38,4.26,3.76,3.08,3.26]]
        # y_test = clf.predict_proba(x_test)
        
        #x_test = [[4648, 4439, 4486, 50.60, 33.30, 51.20]]
        #x_test = [[46.50, 38.70, 62.90, 50.60, 33.30, 51.20],[31.50, 43.60, 57.10, 55.00, 44.50, 3.80]]
        
        #x_test = [["4648","4439","4486","4268","3551","4769"],["3398","4638","4349","3895","3977","4900"]]
        #x_test = [[4648,4439,4486,4268,3551,4769],[3398,4638,4349,3895,3977,4900]]
        
        df_test = pd.DataFrame(columns=feature_name_list[0:len(feature_name_list)-1])
        
        
        #df_test.loc[len(df_test)] = ["4648","4439","4486","4268","3551","4769"];
        #df_test.loc[len(df_test)] = ["3398","4638","4349","3895","3977","4900"];

        #df_test.loc[len(df_test)] = ["4648","4439","4486",50.60, 33.30, 51.20];
        #df_test.loc[len(df_test)] = ["3398","4638","4349",55.00, 44.50, 3.80];
        
        #df_test.loc[len(df_test)] = [46.50, 38.70, 62.90, 50.60, 33.30, 51.20];
        #df_test.loc[len(df_test)] = [31.50, 43.60, 57.10, 55.00, 44.50, 3.80];
        
        del data_type_dict["class"]
        df_test = df_test.astype(dtype=data_type_dict)
        
        df_test.loc[len(df_test)] = [46.50, 38.70, 62.90, 50.60, 33.30, 51.20]
        df_test.loc[len(df_test)] = [31.50, 43.60, 57.10, 55.00, 44.50, 3.80];

        print(model.predict(df_test))
        print(model.predict_proba(df_test))

        # モデル保存
        pickle.dump(model, open(model_filepath, 'wb'))

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
    # argv: {params} {csv_file_path} {eval_file_path} {model_file_path} {feature_name_list} {feature_type_list}
    # ex: boosting_type=gbdt,learning_rate=0.1 D:/Dev/experiment/expr10/arff/9997_rank1.csv D:/Dev/experiment/expr10/evaluation/9997_rank1_eval.txt D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model nw1,nw2,nw3,nw4,nw5,nw6,class float,float,float,float,float,float,category
    
    #param len check
    if len(argv) < 6:
        print('Usage: python xxxModelGenerator.py {params} {csv_file_path} {model_file_path} {feature_name_list} {feature_type_list}')
        return -1
    
    prop = PropertyUtil.getInstance()
    prop.addFile('C:/Dev/workspace/Oxygen/pod_boatrace/properties/expr10/expr10.properties')
    
    # loggin set up
    logSetup()

    trainer = LGBMTrainer()
    trainer.execute(argv[1], argv[2], argv[3], argv[5])
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


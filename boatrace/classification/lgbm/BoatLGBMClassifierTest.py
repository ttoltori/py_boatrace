from datetime import datetime
import json
from logging import getLogger, config
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection._split import train_test_split

from boatrace.common.BoatEnum import DelimiterType
from boatrace.util.PropertyUtil import PropertyUtil
import lightgbm as lgb
import pandas as pd


#
# model生成するクラス
#
class BoatLGBMClassifierTest:
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
        X = df[feature_name_list[0:feature_num-1]]
        y = df[feature_name_list[feature_num-1]]
        
        # train / valid / test (60% / 20% / 20%) に分割
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, shuffle=False)
        
        #モデル파라미터 설정
        model_param_list = param_list_str.split(DelimiterType.DELIM_COMMA.value)
        model_param_dict = {}
        for param in model_param_list:
            key, value = param.split(DelimiterType.DELIM_EQUAL.value)
            # Convert numeric values to int or float
            try:
                if '.' in value:
                    model_param_dict[key] = float(value)
                else:
                    model_param_dict[key] = int(value)
            except ValueError:
                model_param_dict[key] = value
        
        # Debug: print parsed parameters
        print("Model parameters:", model_param_dict)
            
        # 모델 생성
        model = lgb.LGBMClassifier(**model_param_dict)

        # モデル学習
        evals_result = {}
        #model.fit(X_train, y_train)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='logloss',
                 callbacks=[lgb.callback.early_stopping(30), lgb.callback.record_evaluation(evals_result)],)

        # Print the training and validation loss at each boosting round
        #for i, (train_loss, val_loss) in enumerate(zip(evals_result['training']['multi_logloss'], evals_result['valid_1']['multi_logloss'])):
        # for i, (train_loss, val_loss) in enumerate(zip(evals_result['training']['multi_logloss'], evals_result['valid_1']['multi_logloss'])):
        #     if (i+1) % 50 == 0 or i == 0 or i == len(evals_result['training']['multi_logloss']) -1:
        #         print(f"Boosting round {i+1}: training loss = {train_loss:.4f}, validation loss = {val_loss:.4f}")        
        metric_key = next(iter(evals_result['training'].keys()))
        for i, (train_loss, val_loss) in enumerate(zip(evals_result['training'][metric_key], evals_result['valid_1'][metric_key])):
            if i == 0 or i == 49 or i == 99: 
                print(f"Boosting round {i}: training loss = {train_loss:.4f}, validation loss = {val_loss:.4f}")        
        
        y_expected  = y_test
        # y_predicted_proba = model.predict_proba(X_test)
        # print('--- Predicted Probabilities ---')
        # print(y_predicted_proba)

        y_predicted = model.predict(X_test)

        # importance 出力        
        importance = pd.DataFrame(model.feature_importances_, index=model.feature_name_, columns=['importance'])
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(importance)

        # Check for Overfitting
        print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
        print('Validation set score: {:.4f}'.format(model.score(X_valid, y_valid)))
        print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))
        
        # Classification Metrices
        print(classification_report(y_expected, y_predicted))
        
        # モデル保存
        # pickle.dump(model, open(model_filepath, 'wb'))
        # self._logger.info('model created:' + model_filepath)
        
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

    tester = BoatLGBMClassifierTest()
    tester.execute(argv[1], argv[2], argv[3], argv[4], argv[5])
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


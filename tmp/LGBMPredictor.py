import pickle
import sys

from boatrace.server.ModelInfo import ModelInfo
import pandas as pd
from tmp import LGBMPredictor


#
# model生成するクラス
#
class LGBMPreLGBMClassifier():
    DELIM_COMMA = ','
    DELIM_EQUAL = '='
    
    def __init__(self, mi:ModelInfo):
        self._mi_ = mi
        # predictionに必要なfeature, typeを定義しておく
        modelDtype = {}
        for i in range(len(mi.feature_ids)):
            modelDtype[mi.feature_ids[i]] = mi.feature_types[i]
        
        self._df_ = pd.DataFrame(columns=mi.feature_ids).astype(dtype=modelDtype)
    
    # @param param_list_str ex)  boosting_type=gbdt,learning_rate=0.1...
    # @param csv_filepath ex)  D:/Dev/experiment/expr10/arff/9997_rank1.csv
    # @param model_filepath ex)  D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model
    # @param feature_name_list_str ex)  nw1,nw2,nw3,nw4,nw5,nw6,class
    # @param feature_type_list_str ex)  float,float,float,float,float,float,category
    def execute(self, model_filepath:str, feature_value_list:list):
        
        # 모델 로드
        #model = lgb.Booster(model_filepath)
        #model = lgb.LGBMClassifier()
        model = pickle.load(open(model_filepath, 'rb')) 
    
        print( model.predict(feature_value_list) )
        print( model.predict_proba(feature_value_list) );

def main(argv):
    # argv: {params} {csv_file_path} {eval_file_path} {model_file_path} {feature_name_list} {feature_type_list}
    # ex: boosting_type=gbdt,learning_rate=0.1 D:/Dev/experiment/expr10/arff/9997_rank1.csv D:/Dev/experiment/expr10/evaluation/9997_rank1_eval.txt D:/Dev/experiment/expr10/evaluation/model_release/9997/1/nopattern/9997_nopattern_yyyymmdd_rank1.model nw1,nw2,nw3,nw4,nw5,nw6,class float,float,float,float,float,float,category
    
    #param len check
    predictor = LGBMPredictor()
    mode_filepath = "D:/Dev/experiment/expr10/model_release/00005/1/nopattern/00005_nopattern_20160229_rank1.model "
    
    predictor.execute(mode_filepath, [[46.50, 38.70, 62.90, 50.60, 33.30, 51.20],[31.50, 43.60, 57.10, 55.00, 44.50, 3.80]])
    
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))


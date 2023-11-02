import pickle
import sys

import pandas as pd

from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.RemoteRequestParam import RemoteRequestParam
from boatrace.util.PropertyUtil import PropertyUtil
from boatrace.common.BoatConst import BoatConst
from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier
from multiprocessing.dummy import list
from boatrace.common.BoatEnum import FeatureType
from pandas.core.frame import DataFrame
from lightgbm.sklearn import LGBMClassifier
from numpy import ndarray


#
# Classifier of LGBM
#
class BoatLGBMClassifier(AbstractBoatClassifier):
    def __init__(self, mi:ModelInfo) -> None:
        self._mi_:ModelInfo = mi
        self._prop_:PropertyUtil = PropertyUtil.getInstance()
        self._df_:DataFrame
        self._dtype_:dict
        self._model_:LGBMClassifier
        self._isInitialized_:bool = False
    
    def _initialize_(self, param:RemoteRequestParam):
        # predictionに必要なfeature, typeを定義しておく
        self._dtype_ = {}
        for i in range(len(self._mi_.feature_ids)):
            self._dtype_[self._mi_.feature_ids[i]] = self._mi_.feature_types[i]

        self._df_ = pd.DataFrame(columns=self._mi_.feature_ids)
        #self._df_ = pd.DataFrame(columns=self._mi_.feature_ids).astype(dtype=self._dtype_)
        
        # 모델 로드
        model_filepath = self._createModelFilepath(param)
        self._model_ = pickle.load(open(model_filepath, 'rb')) 
    
    def predictProba(self, param:RemoteRequestParam) -> list[float]:
        """
        要求されたpredictionを実行する
        return = propabilities ex) [0.65,0.09,...]
        """
        # 初期化チェック
        if self._isInitialized_ == False:
            self._initialize_(param)
            self._isInitialized_ = True
    
        # if len(self._df_.index) > 0:
            # self._df_ = self._df_.drop(0) # clear data
            #
        # data:list = []
        # # convert param.values to dataframe row
        # for i in range(len(param.values)):
            # if self._mi_.feature_types[i] == FeatureType.FLOAT.value:
                # data.append(float(param.values[i]))
            # else:
                # data.append(param.values[i])
                #
        # self._df_.loc[len(self._df_)] = data # set data
        # self._df_ = self._df_.astype(dtype=self._dtype_)
        
        df = pd.DataFrame([param.values], columns=self._mi_.feature_ids).astype(dtype=self._dtype_)
        
        arr:ndarray = self._model_.predict_proba(df)
        return arr[0].tolist()

    def _createModelFilepath(self, param:RemoteRequestParam) -> str:
        """
        モデルの実体へのfullpathを取得する
        """
        return self._prop_.getProperty('dir_model_release') + \
            '/'.join([param.exNo.zfill(BoatConst.LENGTH_MODEL_NO), param.rankNo, param.pattern, param.modelFileName])
    
from logging import getLogger, Logger
from multiprocessing.dummy import list
import pickle
import sys

from lightgbm.sklearn import LGBMClassifier
from numpy import ndarray

from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier
from boatrace.common.BoatConst import BoatConst
from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.RemoteRequestParam import RemoteRequestParam
from boatrace.util.PropertyUtil import PropertyUtil
import pandas as pd


#
# Classifier of LGBM
#
class BoatLGBMClassifier(AbstractBoatClassifier):
    def __init__(self, mi:ModelInfo) -> None:
        self._mi_:ModelInfo = mi
        self._prop_:PropertyUtil = PropertyUtil.getInstance()
        self._dtype_:dict
        self._model_:LGBMClassifier
        self._isInitialized_:bool = False
        self._logger_:Logger = getLogger('server')
    
    def _initialize_(self, param:RemoteRequestParam):
        # predictionに必要なfeature, typeを定義しておく
        self._dtype_ = {}
        for i in range(len(self._mi_.feature_ids)):
            self._dtype_[self._mi_.feature_ids[i]] = self._mi_.feature_types[i]

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
    
        df = pd.DataFrame([param.values], columns=self._mi_.feature_ids).astype(dtype=self._dtype_)
        
        arr:ndarray = self._model_.predict_proba(df)
        return arr[0].tolist()

    def _createModelFilepath(self, param:RemoteRequestParam) -> str:
        """
        モデルの実体へのfullpathを取得する
        """
        return self._prop_.getProperty('dir_model_release') + \
            '/'.join([param.modelNo.zfill(BoatConst.LENGTH_MODEL_NO), param.rankNo, param.pattern, param.modelFileName])
    
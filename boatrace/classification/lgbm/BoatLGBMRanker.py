from logging import getLogger, Logger
from multiprocessing.dummy import list
import pickle

import numpy as np
from numpy import ndarray

from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier
from boatrace.common.BoatConst import BoatConst
from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.RemoteRequestParam import RemoteRequestParam
from boatrace.util.PropertyUtil import PropertyUtil
import pandas as pd
from boatrace.common.BoatEnum import DelimiterType
from lightgbm.sklearn import LGBMRanker


#
# Classifier of LGBM
#
class BoatLGBMRanker(AbstractBoatClassifier):
    def __init__(self, mi:ModelInfo) -> None:
        self._mi_:ModelInfo = mi
        self._prop_:PropertyUtil = PropertyUtil.getInstance()
        self._dtype_:dict
        self._model_:LGBMRanker
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
    
        #arr2 =  [ [entry1, jyo, raceno...] ... [entry6, jyo, raceno...] ] 
        arr2d= [];
        for entry in param.values:
            arr2d.append(entry.split(DelimiterType.DELIM_COMMA.value))
            
        df = pd.DataFrame(arr2d, columns=self._mi_.feature_ids).astype(dtype=self._dtype_)
        
        arr:ndarray = self._model_.predict(df)
        probabilities = self._ranking_scores_to_probabilities(-arr);
        #probabilities = 1 - (1 / (1 + np.exp(-arr)))
        
        
        return probabilities.tolist();

    def _createModelFilepath(self, param:RemoteRequestParam) -> str:
        """
        モデルの実体へのfullpathを取得する
        """
        return self._prop_.getProperty('dir_model_release') + \
            '/'.join([param.modelNo.zfill(BoatConst.LENGTH_MODEL_NO), param.rankNo, param.pattern, param.modelFileName])

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
    
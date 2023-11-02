from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier
from boatrace.common import BoatEnum
from boatrace.factory.BoatClassifierFactory import BoatClassifierFactory,\
    AbstractBoatClassifierFactory
from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.RemoteRequest import RemoteRequest
from boatrace.server.RemoteRequestParam import RemoteRequestParam
from boatrace.server.RemoteResponse import RemoteResponse
from boatrace.server.service.ClassifierQueue import ClassifierQueue
from boatrace.util.PropertyUtil import PropertyUtil
from boatrace.util.Singleton import Singleton
from boatrace.server.service.AbstractService import AbstractService


class ClassificationLGBMService(AbstractService, Singleton):
    """
    lightGBMのclassification, regressionを提供するクラス
    """
    def __init__(self):
        self._prop_:PropertyUtil = PropertyUtil.getInstance()
        
        # classifier factory
        self._factory_:AbstractBoatClassifierFactory = BoatClassifierFactory()
        
        # initialize queue
        queue_max:int = int(self._prop_.getProperty('classifier_queue_max'))
        
        self._queue_ = ClassifierQueue(queue_max)
        
    def execute(self, req:RemoteRequest) -> RemoteResponse:
        param:RemoteRequestParam = req.param
        
        # ML実行
        values:list[float] = self._classify_(param)
        
        return RemoteResponse(req.id, req.algorithmId, values, BoatEnum.ServiceStatus.OK.value)
        
    def _classify_(self, param:RemoteRequestParam) -> list[float]:
        # モデルファイル情報を取得
        mi:ModelInfo = self._prop_.getModelInfo(param.modelNo, param.rankNo)
        
        # classifier取得 key = filename
        # ex) 1_nopattern_20151231_rank1.model
        clf:AbstractBoatClassifier = self._queue_.get(param.modelFileName)
        
        # classifierがキューに存在しない場合
        if clf == None:
            # モデルファイル情報からclassifierを生成する
            clf = self._factory_.create(mi)
            
            # queueに登録する
            self._queue_.add(param.modelFileName, clf)
        
        return clf.predictProba(param)

from logging import Logger, getLogger, StreamHandler, Formatter, DEBUG
from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier
from boatrace.common.BoatEnum import ServiceType
from boatrace.factory.BoatClassifierFactory import AbstractBoatClassifierFactory
from boatrace.factory.ServiceFactory import AbstractServiceFactory
from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.RemoteRequest import RemoteRequest
from boatrace.server.RemoteRequestParam import RemoteRequestParam
from boatrace.server.RemoteResponse import RemoteResponse
from boatrace.server.service.AbstractService import AbstractService
from boatrace.util.Singleton import Singleton


class TestUtil():
    @staticmethod
    def createRemoteRequestParam(exNo:str, modelNo:str, rankNo:str, pattern:str, ymd:str, modelFileName, values:list[str]) -> RemoteRequestParam:
        cReq:RemoteRequestParam = RemoteRequestParam()
        cReq.values = values
        cReq.exNo = exNo
        cReq.modelNo = modelNo
        cReq.rankNo = rankNo
        cReq.pattern = pattern
        cReq.ymd = ymd
        cReq.modelFileName = modelFileName
        
        return cReq

    @staticmethod
    def createRemoteRequest(exNo:str, modelNo:str, rankNo:str, pattern:str, ymd:str, modelFileName, values:list[str]) -> RemoteRequest:
        cReq:RemoteRequest = RemoteRequest()
        cReq.algorithmId = 'cf_lgbm_py'
        cReq.id = 'id1'
        param = TestUtil.createRemoteRequestParam(exNo, modelNo, rankNo, pattern, ymd, modelFileName, values)
        cReq.param = param
        
        return cReq

    @staticmethod
    def getConsoleLogger() -> Logger:
        formatter = Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
        handler = StreamHandler()
        handler.setFormatter(formatter)
        
        rootLogger:Logger = getLogger()
        rootLogger.addHandler(handler)
        rootLogger.setLevel(DEBUG)
        
        return rootLogger

class MockBoatClassifierFactory(AbstractBoatClassifierFactory):
    def __init__(self):
        self._classifier_:AbstractBoatClassifier;
        
    def create(self, mi:ModelInfo) -> AbstractBoatClassifier:
        return self._classifier_
    
    def setReturn(self, classifier:AbstractBoatClassifier) -> None:
        self._classifier_ = classifier


class MockBoatLGBMClassifier(AbstractBoatClassifier):
    def __init__(self, mi:ModelInfo) -> None:
        self._values_:list[float]
        
    def predictProba(self, req:RemoteRequestParam) -> list[float]:
        return self._values_
    
    def setReturn(self, values:list[float]) -> None:
        self._values_ = values

class MockServiceFactory(AbstractServiceFactory):
    def __init__(self):
        self._service_:AbstractService
        
    def create(self, service_type:ServiceType) -> AbstractService:
        return self._service_
    
    def setReturn(self, service:AbstractService) -> None:
        self._service_ = service


class MockClassificationLGBMService(AbstractService, Singleton):
    def __init__(self) -> None:
        self._res_:RemoteResponse
        
    def execute(self, req:RemoteRequest) -> RemoteResponse:
        return self._res_
    
    def setReturn(self, res:RemoteResponse) -> None:
        self._res_ = res
        
        
        

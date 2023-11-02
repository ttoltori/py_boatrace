from builtins import Exception
from json.decoder import JSONObject
from logging import getLogger

from boatrace.common.BoatEnum import ServiceType
from boatrace.factory.ServiceFactory import ServiceFactory, AbstractServiceFactory
from boatrace.server.RemoteRequest import RemoteRequest
from boatrace.server.RemoteResponse import RemoteResponse
from boatrace.server.service.AbstractRequestDispatcher import AbstractRequestDispatcher
from boatrace.server.service.AbstractService import AbstractService
from boatrace.util.JsonUtil import JsonUtil


class JsonRequestDispatcher(AbstractRequestDispatcher):
    def __init__(self):
        self._service_factory_:AbstractServiceFactory = ServiceFactory()
        # classification service map
        self._map_service_:dict[str, AbstractService] = {}
        self._logger_ = getLogger('server')
        self._is_initialized_:bool = False
    
    def _initialize_(self) -> None:
        for e in ServiceType:
                self._map_service_[e.value] = self._service_factory_.create(e)
    
    def _ensureInitialized_(self):
        if not self._is_initialized_ :
            self._initialize_()
            self._is_initialized_ = True            
        
    def dispatch(self, jsonStr:str) -> str:
        """
        stringの要求をjson parsingしてサービスへdispatchする
        """
        self._ensureInitialized_() #初期化確認
        
        resultObject:object # 結果object
        
        json:JSONObject
        try:
            json = JsonUtil.decode(jsonStr)
        except Exception:
            self._logger_.error('json decode failed.')
            return "Error:json decode failed."
        
        # jsonのalgorithmIdからMLの種類を判定して適正なサービスへ受け渡す
        # ex) cf_lgbm-1_py
        algorithm_id:str = json['algorithmId'] 
        #classification or regression
        if (algorithm_id.startswith('cf_')) or (algorithm_id.startswith('rg_')):
            req:RemoteRequest
            try: 
                req = JsonUtil.decodeGenericObject(jsonStr)
            except Exception:
                self._logger_.error('json decodeGenericObject failed.')
                return "Error:json decodeGenericObject failed."
                
            resultObject = self._dispatchClassification_(req)
        else:
            return "Error:unsupported algorithm. " + algorithm_id
        
        # 結果objectをjson文字列へencodeしてreturnする
        res:str
        try:
            res = JsonUtil.encodeCustomObject(resultObject)
        except Exception:
            self._logger_.error('json encodeCustomObject failed.')
            return "Error:json encodeCustomObject failed.";
        
        return res
        
    def _dispatchClassification_(self, req:RemoteRequest) -> RemoteResponse:
        """
        RemoteRequestでclassification(regreggion)を実行してRemoteResponseを返却する
        """
        # algorithmidをserviceTypeへ変換して該当のserviceを取得する
        # ex) cf_bayes-1_wk -> cf_bayes_wk
        service_type_str:str = self._getServiceTypeString_(req.algorithmId)
        service:AbstractService = self._map_service_[service_type_str]
        if service == None:
            return RemoteResponse(req.id, req.algorithmId, [], 'Service not exist')
        try:
            # service 実行
            return service.execute(req)
        except Exception:
            self._logger_.exception('service execution failed.')
            return RemoteResponse(req.id, req.algorithmId, [], 'service execution failed.')
        
    def _getServiceTypeString_(self, algorithm_id:str):
        """
        algorithmidをserviceTypeへ変換する
        ex) cf_bayes-1_py -> cf_bayes_py
        """
        token:list[str] = algorithm_id.split('_')
        serviteType:str = '_'.join([ token[0],  token[1].split('-')[0], token[len(token)-1]])
        
        return serviteType
        
    
        
    
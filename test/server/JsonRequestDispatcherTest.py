from unittest import TestCase

from boatrace.common.BoatEnum import ServiceStatus
from boatrace.server.RemoteResponse import RemoteResponse
from boatrace.server.service.JsonRequestDispatcher import JsonRequestDispatcher
from boatrace.util.JsonUtil import JsonUtil
from test.TestUtil import MockClassificationLGBMService, MockServiceFactory


class JsonRequestDispatcherTest(TestCase):

    def setUp(self):
        self.target:JsonRequestDispatcher = JsonRequestDispatcher()
        pass
    
    def _setupMock(self):
        mockService:MockClassificationLGBMService = MockClassificationLGBMService()
        expectedRes:RemoteResponse = RemoteResponse("id1", 'cf_lgbm-1_py', [1.0,2.0,3.0,4.0,5.0,6.0], ServiceStatus.OK.value)
        mockService.setReturn(expectedRes)
        
        mockSrvFactory:MockServiceFactory = MockServiceFactory()
        mockSrvFactory.setReturn(mockService)
        self.target._service_factory_ = mockSrvFactory
        
    def test_dispatch_OK(self):
        # 準備
        self._setupMock()
        expectedRes:RemoteResponse = RemoteResponse("id1", 'cf_lgbm-1_py', [1.0,2.0,3.0,4.0,5.0,6.0], ServiceStatus.OK.value)
        
        # 実行
        resStr:str
        reqStr:str = '{"algorithmId": "cf_lgbm_py", "id": "id1", "param": {"values": [], "exNo": "1", "modelNo": "1", "rankNo": "1", "pattern": "nopattern", "ymd": "20170101", "modelFileName": "00001_nopattern_20151231_rank1.model"}}'
        resStr = self.target.dispatch(reqStr)
        
        reqStr:str = '{"algorithmId": "rg_lgbm_py", "id": "id1", "param": {"values": [], "exNo": "1", "modelNo": "1", "rankNo": "1", "pattern": "nopattern", "ymd": "20170101", "modelFileName": "00001_nopattern_20151231_rank1.model"}}'
        resStr = self.target.dispatch(reqStr)

        resObject:RemoteResponse = JsonUtil.decodeGenericObject(resStr)
        self.assertNotEqual(resStr, None)
        self.assertEqual(expectedRes.id, resObject.id)
        self.assertEqual(expectedRes.algorithmId, resObject.algorithmId)
        self.assertEqual(expectedRes.values, resObject.values)
    
    def test_dispatch_json_decode_NG(self):
        # 準備
        self._setupMock()
        
        # 実行
        reqStr:str = '{"algorithmId": "cf_lgbm_py, "id": "id1", "param": {"values": [], "exNo": "1", "modelNo": "1", "rankNo": "1", "pattern": "nopattern", "ymd": "20170101", "modelFileName": "00001_nopattern_20151231_rank1.model"}}'
        resStr:str = self.target.dispatch(reqStr)
        
        self.assertEqual(resStr, 'Error:json decode failed.')

    def tearDown(self):
        pass
        
    
        
        

from unittest import TestCase

from boatrace.util.JsonUtil import JsonUtil
from boatrace.server.RemoteRequestParam import RemoteRequestParam
from boatrace.server.RemoteRequest import RemoteRequest
from json.decoder import JSONObject
from test.TestUtil import TestUtil

class JsonUtilTest(TestCase):

    def setUp(self):
        pass
        
    def test_encode(self):
        data:dict = {"values": ["sadfsdf", 1.34] }
        
        result:str = JsonUtil.encode(data)
        
        self.assertNotEqual(result, None)


    def test_decode(self):
        jsonStr:str = '{"values": ["sadfsdf", 1.34]}'
        
        result:JSONObject = JsonUtil.decode(jsonStr)
        
        self.assertNotEqual(result, None)
        
        
    def test_encodeCustomObject_OK(self):
        req:RemoteRequest = TestUtil.createRemoteRequest('1', '1', '1', 'nopattern', '20160201', 'testfile.model', ['4045', 3.0])
        
        result:str = JsonUtil.encodeCustomObject(req)
        
        self.assertEqual(len(result) > 0, True)

    def test_decodeGenericObject_OK(self):
        jsonStr:str = '{"algorithmId": "cf_lgbm_py", "id": "id1", "param": {"values": ["4045", 3.0], "exNo": "1", "modelNo": "1", "rankNo": "1", "pattern": "nopattern", "ymd": "20160201", "modelFileName": "testfile.model"}}'
        
        result:RemoteRequest = JsonUtil.decodeGenericObject(jsonStr)
        param:RemoteRequestParam = result.param
        
        self.assertNotEqual(result, None)
        self.assertEqual(result.algorithmId, 'cf_lgbm_py')
        self.assertEqual(param.modelNo, '1')
    
    def tearDown(self):
        pass

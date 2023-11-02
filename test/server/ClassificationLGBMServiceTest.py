from unittest import TestCase

from boatrace.server import RemoteResponse
from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.RemoteRequest import RemoteRequest
from boatrace.server.service.ClassificationLGBMService import ClassificationLGBMService
from boatrace.util.PropertyUtil import PropertyUtil
from test.TestUtil import TestUtil, MockBoatClassifierFactory, \
    MockBoatLGBMClassifier
from boatrace.server.service.AbstractService import AbstractService


class ClassificationLGBMServiceTest(TestCase):
    def _setupProperty_(self):
        self.prop = PropertyUtil.getInstance()
        # 1_rank1=r1-123456::cf_lgbm-1_py::nw1,nw2,nw3,nw4,nw5,nw6::float,float,float,float,float,float
        self.prop.putProperty('1_rank1', 'r1-123456::cf_lgbm-1_py::nw1,nw2,nw3,nw4,nw5,nw6::float,float,float,float,float,float')
        self.prop.putProperty('classifier_queue_max', '30')
        
    
    def setUp(self):
        self._setupProperty_()
        self.target:AbstractService = ClassificationLGBMService()
        
    def test_execute_empty_queue_OK(self):
        mockClassifier:MockBoatLGBMClassifier = MockBoatLGBMClassifier(ModelInfo())
        mockFactory:MockBoatClassifierFactory = MockBoatClassifierFactory()
        mockClassifier.setReturn([1.0,2.0,3.0,4.0,5.0,6.0])
        mockFactory.setReturn(mockClassifier)
        self.target._factory_ = mockFactory
        
        req:RemoteRequest = TestUtil.createRemoteRequest('1', '1', '1', 'nopattern', '20170101', '00001_nopattern_20151231_rank1.model', [])
        
        result:RemoteResponse = self.target.execute(req)
        
        self.assertNotEqual(result, None)
    
    def test_execute_notempty_queue_OK(self):
        mockClassifier:MockBoatLGBMClassifier = MockBoatLGBMClassifier(ModelInfo())
        mockClassifier.setReturn([1.0,2.0,3.0,4.0,5.0,6.0])
        self.target._queue_.add('00001_nopattern_20151231_rank1.model', mockClassifier)
        
        req:RemoteRequest = TestUtil.createRemoteRequest('1', '1', '1', 'nopattern', '20170101', '00001_nopattern_20151231_rank1.model', [])
        
        res:RemoteResponse = self.target.execute(req)
        
        self.assertNotEqual(res, None)
        self.assertEqual(res.id, 'id1')
        self.assertEqual(res.values[5], 6)

    def tearDown(self):
        pass
        
    
        
        

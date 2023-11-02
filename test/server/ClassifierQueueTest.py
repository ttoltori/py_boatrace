from unittest import TestCase

from boatrace.server.ModelInfo import ModelInfo
from boatrace.server.service.ClassifierQueue import ClassifierQueue
from test.TestUtil import MockBoatLGBMClassifier, TestUtil

class ClassifierQueueTest(TestCase):

    def setUp(self):
        self.target:ClassifierQueue = ClassifierQueue(1)
        self.target._logger_ = TestUtil.getConsoleLogger()
        
    def test_add_OK(self):
        clf1 = MockBoatLGBMClassifier(ModelInfo())
        
        self.assertEqual(self.target.size(), 0)
        self.target.add("key1", clf1)
        self.assertEqual(self.target.size(), 1)
        
    def test_add_over_max_OK(self):
        clf1 = MockBoatLGBMClassifier(ModelInfo())
        clf2 = MockBoatLGBMClassifier(ModelInfo())
        
        self.assertEqual(self.target.size(), 0)
        self.target.add("key1", clf1)
        self.target.add("key2", clf2)
        self.assertEqual(self.target.size(), 1)
    
    def test_get_OK(self):
        clf1 = MockBoatLGBMClassifier(ModelInfo())
        clf2 = MockBoatLGBMClassifier(ModelInfo())
        
        self.target.add("key1", clf1)
        self.target.add("key2", clf2)
        self.assertEqual(self.target.size(), 1)
        
        res = self.target.get('key2')
        
        self.assertNotEqual(res, None)
        
    def tearDown(self):
        pass
        
    
        
        

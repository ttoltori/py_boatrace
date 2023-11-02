from unittest import TestCase

from boatrace.util.JsonUtil import JsonUtil
import test.TestUtil
from boatrace.server.service.ClassifierQueue import ClassifierQueue

class TemplateTest(TestCase):

    def setUp(self):
        self.target = None
        
        pass
        
    def test_template_method(self):
        data:dict = {"values": ["sadfsdf", 1.34] }
        
        result:str = JsonUtil.encode(data)
        
        self.assertNotEqual(result, None)
    
    def tearDown(self):
        pass
        
    
        
        

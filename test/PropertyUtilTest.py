from unittest import TestCase

from boatrace.util.PropertyUtil import PropertyUtil


class PropertyUtilTest(TestCase):
    def setUp(self):
        self.target = PropertyUtil()
        
    def test_addFile_OK(self):
        self.target.addFile('C:/Dev/workspace/Oxygen/pod_boatrace/properties/expr10/expr10.properties')
        result = self.target.getProperty('file_python_log_config')
        
        self.assertEqual(len(result) > 0, True)
    
    def test_getModelInfo_OK(self):
        self.target.addFile('C:/Dev/workspace/Oxygen/pod_boatrace/properties/expr10/model.properties')
        result = self.target.getModelInfo('1', 'rank1')
    
        self.assertNotEqual(result, None)
    def tearDown(self):
        TestCase.tearDown(self)
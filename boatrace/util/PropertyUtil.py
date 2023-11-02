from boatrace.server.ModelInfo import ModelInfo
from boatrace.util.Properties import Properties
from boatrace.util.Singleton import Singleton
from multiprocessing.dummy import list
from jsonschema._validators import properties
#
# java properties access用singleton class
#
class PropertyUtil(Singleton):
    def __init__(self):
        self.__prop__ = Properties()
        self._flieList_:list = []
    def getModelInfo(self, model_no, rank_no):
        """
        model_no, rank_noをキーにModelInfoを取取得する
        """
        key = '_'.join([model_no, 'rank' + rank_no]) 
        value = self.__prop__.getProperty(key)
        token = value.split('::')
        
        mi = ModelInfo()
        mi.class_id = token[0];
        mi.algorithm_id = token[1]; 
        mi.feature_ids = token[2].split(',')
        mi.feature_types =  token[3].split(',')
        
        return mi
    
    def reload(self):
        prop:properties = Properties()
        for filepath in self._flieList_:
            prop.load(filepath, '=', '#')
    
        self.__prop__ = prop
    def addFile(self, filepath):
        #load properties file
        self.__prop__.load(filepath, '=', '#')
        self._flieList_.append(filepath)
    
    def getProperty(self, key) -> str:
        return self.__prop__.getProperty(key)
    
    def putProperty(self, key, value):
        self.__prop__.putProperty(key, value)
    
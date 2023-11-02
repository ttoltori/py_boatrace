import json
from json.decoder import JSONObject
from json.encoder import JSONEncoder


class JsonUtil():
    """
    provide json string <-> object functionality
    string <-> python object
    string <-> custom object
    """
    @staticmethod
    def encode(obj:object) -> str:
        '''
        convert python object to string
        '''
        return json.dumps(obj)

    @staticmethod
    def decode(jsonStr:str) -> JSONObject:
        '''
        convert string to python object 
        '''
        return json.loads(jsonStr)

    @staticmethod
    def encodeCustomObject(obj:object) -> str:
        '''
        convert custom object to string
        '''
        return json.dumps(obj, cls=GenericEncoder)
    
    @staticmethod
    def decodeGenericObject(jsonStr:str) -> object:
        """
        convert json string to generic object.
        refer to JsonUtilTest.py for usage.
        """
        return json.loads(jsonStr, object_hook=GenericObject.from_dict)

class GenericObject:
    @classmethod
    def from_dict(cls, dt) -> object:
        obj = cls()
        obj.__dict__.update(dt)
        
        return obj
    
class GenericEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
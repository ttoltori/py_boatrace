#
# javaのpropertiesファイルを読み込む
#

class Properties():
    def __init__(self):
        self.__prop__ = {}

    def load(self, filepath, sep='=', comment_char='#'):
        """
        propertiesファイルをロードする
        """
        with open(filepath, "rt", encoding='utf-8') as f:
            for line in f:
                l = line.strip()
                if l and not l.startswith(comment_char):
                    key_value = l.split(sep)
                    key = key_value[0].strip()
                    value = sep.join(key_value[1:]).strip().strip('"')
                    self.__prop__[key] = value 
    
    def getProperty(self, key) -> str:
        return self.__prop__.get(key)
    
    def putProperty(self, key, value):
        self.__prop__[key] = value;
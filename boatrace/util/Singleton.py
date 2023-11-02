class Singleton(object):
    @classmethod
    def getInstance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

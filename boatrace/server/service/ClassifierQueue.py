from logging import getLogger, Logger

from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier

class ClassifierQueue():
    def __init__(self, queue_max:int):
        self._queue_max_ = queue_max
        self._logger_:Logger = getLogger('server')
        
        # item map  ex) key = 1_nopattern_20151231_rank1.model  value=Classifier implementation 
        self._map_item_:dict[str, AbstractBoatClassifier] = {}
        
        #item key list
        self._list_key_:list[str] = []
        
    def add(self, key:str, clf: AbstractBoatClassifier) -> None:
        """
        add item
        
        """
        # maxに達した
        if len(self._list_key_) >= self._queue_max_ :
            self._destroyOld_()
        
        # itemをmap, listに追加
        self._map_item_[key] = clf
        self._list_key_.append(key)

        self._logger_.debug('item added...queue size={0}, remove item={1}'
                            .format(len(self._list_key_), key) )
        
    def size(self) -> int:
        """
        return size of the map
        """
        return len(self._map_item_)
    
    def get(self, key:str) -> AbstractBoatClassifier:
        """
        get item
        """
        return self._map_item_.get(key, None)
    
    def _destroyOld_(self) -> None:
        key:str = self._list_key_[0] 
        
        # mapItemから除去
        del self._map_item_[self._list_key_[0]]
        
        del self._list_key_[0]
        
        self._logger_.debug('item removed...queue size={0}, remove item={1}'.
                            format(len(self._list_key_), key ))

#
# model情報クラス。
# Predictorをロードする時のDataFrame生成時使う。
#
class ModelInfo:
    def __init__(self):
        self.class_id: str
        self.algorithm_id: str
        self.feature_ids = list[str]
        self.feature_types = list[str]
        

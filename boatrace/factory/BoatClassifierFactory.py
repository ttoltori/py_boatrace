from boatrace.server.ModelInfo import ModelInfo
from boatrace.classification.lgbm.AbstractBoatClassifier import AbstractBoatClassifier
from boatrace.classification.lgbm.BoatLGBMClassifier import BoatLGBMClassifier
from boatrace.classification.lgbm.BoatLGBMRegressor import BoatLGBMRegressor

class AbstractBoatClassifierFactory():
    """
    factory interface
    """
    def create(self, mi:ModelInfo) -> AbstractBoatClassifier:
        raise NotImplementedError


class BoatClassifierFactory(AbstractBoatClassifierFactory):
    """
    factory of AbstractBoatClassifier
    """

    def create(self, mi:ModelInfo) -> AbstractBoatClassifier:
        if mi.algorithm_id.startswith('cf_lgbm'):
            return BoatLGBMClassifier(mi)  # LGBMClassifier
        elif mi.algorithm_id.startswith('rg_lgbm'):
            return BoatLGBMRegressor(mi) #LGBMRegressor


from boatrace.common.BoatEnum import ServiceType
from boatrace.server.service.ClassificationService import ClassificationService

class AbstractServiceFactory():
    """
    factory interface
    """
    def create(self, service_type:ServiceType) -> object:
        raise NotImplementedError

    
class ServiceFactory():
    """
    service instance creation
    """

    def create(self, service_type:ServiceType) -> object:
        if ((service_type == ServiceType.CF_LGBM_PY) or  
                (service_type == ServiceType.RG_LGBM_PY) or 
                (service_type == ServiceType.RK_CATB_PY) ): 
            # LGBM classification, regression
            return ClassificationService.getInstance()
        else:
            return None

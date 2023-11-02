from enum import Enum


class DelimiterType(Enum):
    DELIM_COMMA = ','
    DELIM_EQUAL = '='

class ServiceType(Enum):
    CF_LGBM_PY = 'cf_lgbm_py' # Classification of  LightGBM
    RG_LGBM_PY = 'rg_lgbm_py' # Regression of LightGBM 

class ServiceStatus(Enum):
    OK = "OK" # OK
    NG = 'NG' # NG

class FeatureType(Enum):
    FLOAT = 'float'
    CATEGORY = 'category'    
# usage: DemiliterType.DELIM_COMMA.name or value
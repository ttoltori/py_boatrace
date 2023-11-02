from boatrace.server.RemoteResponse import RemoteResponse
from boatrace.server.RemoteRequest import RemoteRequest


class AbstractService():
    """
    classification or regression interface
    """
    def execute(self, req:RemoteRequest) -> RemoteResponse:
        raise NotImplementedError 
        

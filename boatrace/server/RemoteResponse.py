class RemoteResponse:
    def __init__(self, idd:str, algorithmId:str, values:list[float], status:str):
        self.id: str = idd
        self.algorithmId: str = algorithmId
        self.values: list[float] = values
        self.status:str = status
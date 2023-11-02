
class AbstractRequestDispatcher():
    """
    string requset to service dispatcher interface
    """
    def dispatch(self, jsonStr: str) -> str:
        raise NotImplementedError

from boatrace.server.RemoteRequestParam import RemoteRequestParam
class AbstractBoatClassifier():
    """
    classifier or regressorの共通interface
    """
    def predictProba(self, param: RemoteRequestParam) -> list[float]:
        """
        classificationのreturn -> [float,...n]  n=class valuesの数
        regressionのreturn -> [0] [0]番目のみに予測値を格納
        """
        raise NotImplementedError
        
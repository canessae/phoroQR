import json
import numpy as np
from types import SimpleNamespace

class Features:

    position = [0, 0]

    def __init__(self, feat = None):
        if feat is not None:
            self.position = feat

class EmbeddedData:
    hash = ""
    features = Features()

    def __init__(self, hash = None, feat = None):
        if hash is not None:
            self.hash_code = hash
        if feat is not None:
            self.features = feat

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def setFeatures(self, feat):
        self.features = feat

    def setHash(self, hash):
        self.hash = hash

    @staticmethod
    def fromJson(data):
        return json.loads(data, object_hook=lambda d: SimpleNamespace(**d))

#test
if __name__ == "__main__":
    tt = [1, 2]
    test = EmbeddedData("ciao", Features(tt))
    jsonData = json.dumps(test.toJson(), indent=2)
    print(jsonData)
    print(type(jsonData))
    decoded = json.loads(jsonData)
    print(decoded)
    print(type(decoded))

    print(">>\n\n")
    x = EmbeddedData.fromJson(decoded)
    print(x)

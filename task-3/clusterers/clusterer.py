class Clusterer:
    model = None
    data = None

    def __init__(self, data):
        self.data = data

    def fit_predict(self):
        return self.model.fit_predict(self.data)

import numpy

class regress():
    def __init__(self, regressor):
        self.regressor = regressor

    def add_training_data(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_X_mean = numpy.mean(train_X, 0)
        self.train_X_std = numpy.std(train_X, 0)
        self.train_Y_mean = numpy.mean(train_Y)
        self.train_Y_std = numpy.std(train_Y)
        self.train_X_scaled = (train_X - self.train_X_mean) / self.train_X_std
        self.train_Y_scaled = (train_Y - self.train_Y_mean) / self.train_Y_std

    def train(self):
        self.fit = self.regressor.fit(self.train_X_scaled, self.train_Y_scaled)

    def predict(self, test_X):
        out_scaled = self.regressor.predict((test_X - self.train_X_mean)/self.train_X_std)
        out = out_scaled * self.train_Y_std + self.train_Y_mean
        return out

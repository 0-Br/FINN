from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 自己补的话，在train里再补一下
class Critic:
    """"""

    def __init__(self):
        """"""
        self.num = 0
        self.trues = []
        self.preds = []
        self.r2 = None
        self.mse = None
        self.mae = None

    def clear(self):
        """"""
        self.num = 0
        self.trues = []
        self.preds = []
        self.r2 = None
        self.mse = None
        self.mae = None

    def record(self, trues, preds):
        """"""
        for i in range(len(trues)):
            self.num += 1
            self.trues.append(trues[i])
            self.preds.append(preds[i])

    def judge(self):
        """"""
        self.r2 = r2_score(self.trues, self.preds)
        self.mse = mean_squared_error(self.trues, self.preds)
        self.mae = mean_absolute_error(self.trues, self.preds)

    def print(self):
        """"""
        print(f"r2: {self.r2}")
        print(f"mse:       {self.mse}")
        print(f"mae:       {self.mae}")

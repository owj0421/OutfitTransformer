from sklearn.metrics import roc_auc_score, accuracy_score


class MetricCalculator:
    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.y_score = []
        
    def update(self, y_true, y_pred, y_score=[]):
        self.y_true += y_true
        self.y_pred += y_pred
        self.y_score += y_score

    def calc_auc(self):
        return roc_auc_score(self.y_true, self.y_score)
    
    def calc_acc(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def clean(self):
        self.y_true = []
        self.y_pred = []
        self.y_score = []
import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!' # thanks
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indices = self.indices_list[bag]
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag))
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        predictions = np.mean([model.predict(data) for model in self.models_list], axis=0)
        return predictions
    
    def _get_oob_predictions_from_every_model(self):
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        for idx, (data_point, target_value) in enumerate(zip(self.data, self.target)):
            for model, indices in zip(self.models_list, self.indices_list):
                if idx not in indices:
                    prediction = model.predict(data_point.reshape(1, -1))
                    list_of_predictions_lists[idx].append(prediction)
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([np.mean(preds) if len(preds) > 0 else None for preds in self.list_of_predictions_lists])
  
    #def OOB_score(self):
    #    '''
    #    Compute mean square error for all objects, which have at least one prediction
    #    '''
    #    self._get_averaged_oob_predictions()
    #    return # Your Code Here
    def OOB_score(self):
        self._get_averaged_oob_predictions()
        valid_targets = [target for target, preds in zip(self.target, self.list_of_predictions_lists) if len(preds) > 0]
        valid_predictions = [pred for pred in self.oob_predictions if pred is not None]
        mse = np.mean((np.array(valid_targets) - np.array(valid_predictions)) ** 2)
        return mse

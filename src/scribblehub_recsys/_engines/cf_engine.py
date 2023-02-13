import os
import itertools
import copy
import logging
import time
from datetime import datetime

from implicit import evaluation
from implicit.cpu.als import AlternatingLeastSquares


class CFRecommender(AlternatingLeastSquares):

    def __init__(
        self, 
        alpha=20, 
        regularization=10, 
        factors=80, 
        iterations=25
    ):
        super().__init__(alpha=alpha, 
                         regularization=regularization,
                         factors=factors,
                         iterations=iterations)
        
        self._start_time = ""


    def _set_start_time(self):
        self._start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        
    def _get_start_time(self):
        return self._start_time


    def _print_log(self, row, header=False, spacing=12):
        top = ''
        middle = ''
        bottom = ''
        for r in row:
            top += '+{}'.format('-'*spacing)
            if isinstance(r, str):
                middle += '| {0:^{1}} '.format(r, spacing-2)
            elif isinstance(r, int):
                middle += '| {0:^{1}} '.format(r, spacing-2)
            elif isinstance(r, float):
                middle += '| {0:^{1}.5f} '.format(r, spacing-2)
            bottom += '+{}'.format('='*spacing)
        top += '+'
        middle += '|'
        bottom += '+'
        if header:
            print(top)
            print(middle)
            print(bottom)
            with open(os.path.join(os.path.dirname("__file__"),
                                            "results/grid_search_" + 
                                            self._get_start_time() + 
                                            ".log"), 'a') as output_file:
                output_file.write(top +'\n')
                output_file.write(middle + '\n')
                output_file.write(bottom + '\n')
        else:
            print(middle)
            print(top)
            with open(os.path.join(os.path.dirname("__file__"), 
                                            "results/grid_search_" + self._get_start_time() + ".log"), 'a') as output_file:
                output_file.write(middle + '\n')
                output_file.write(top + '\n')
        
            
    def _learning_curve(self, this_model, train, test, epochs, user_index):
        
        if not user_index:
            user_index = range(train.shape[0])
        prev_epoch = 0
        model_precision = []
        model_map = []
        model_ndcg = []
        
        headers = ['epochs', 'p@k', 'map@k',
                'ndcg@k']
        self._print_log(headers, header=True)
        
        for epoch in epochs:
            this_model.iterations = epoch - prev_epoch
            if not hasattr(this_model, 'user_vectors'):
                this_model.fit(train, show_progress=False)
            else:
                print("I'm not supposed to be here.")
                this_model.fit_partial(train, show_progress=False)
            model_precision.append(evaluation.precision_at_k(this_model, train, test, show_progress=False))
            model_map.append(evaluation.mean_average_precision_at_k(this_model, train, test, show_progress=False))
            model_ndcg.append(evaluation.ndcg_at_k(this_model, train, test, show_progress=False))
            row = [epoch, model_precision[-1],
                model_map[-1], model_ndcg[-1]]
            self._print_log(row)
            prev_epoch = epoch
        return this_model, model_precision, model_map, model_ndcg

    def _grid_search_learning_curve(self, train, test, param_grid,
                                user_index=None, epochs=range(2, 40, 2)):
        """
        "Inspired" (stolen) from sklearn gridsearch
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py
        """
        base_model = AlternatingLeastSquares()
        curves = []
        keys, values = zip(*param_grid.items())
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            this_model = copy.deepcopy(base_model)
            print_line = []
            
            for k, v in params.items():
                setattr(this_model, k, v)
                print_line.append((k, v))
                
            print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
            with open(os.path.join(os.path.dirname("__file__"),
                                   "results/grid_search_" + 
                                   self._get_start_time() + 
                                   ".log"), 'a') as output_file:
                
                output_file.write(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line) + '\n')
                output_file.close()
            
            start = time.time()
            _, patk, mapatk, ndcgatk = self._learning_curve(this_model, train, test,
                                                    epochs, user_index=user_index)
            logging.info("Completed iteration in {} seconds.", time.time() - start)
            curves.append({'params': params,
                        'patk': patk,
                        'mapatk': mapatk,
                        'ndcgatk':ndcgatk})
        return curves


    def grid_search(self, data, param_grid=None):
        
        train, test = evaluation.train_test_split(data)
        
        if not param_grid:
            param_grid = {'factors': [40, 60, 80, 100, 120], #The overall weight given to user's rating to an item. Must not be 0.
                        'regularization': [1, 5, 10, 15, 20, 25], #regularization discourages learning a more complex or flexible model, so as to avoid the risk of overfitting.
                        'alpha': [5, 10, 15, 20, 25]} #The learning rate parameter of the algorithm. Higher alpha gives bigger "jumps". Must not be 0.
        self._set_start_time()
        
        with open(os.path.join(os.path.dirname("__file__"),
                               "results/grid_search_" + 
                               self._get_start_time() + 
                               ".log"), 'w') as output_file:
            
            output_file.write("Starting new Grid Search...\n")
        
        curves = self._grid_search_learning_curve(train, test,
                                            param_grid)

        best_curves = sorted(curves, key=lambda x: max(x['patk']), reverse=True)
                            
        print(best_curves[0]['params'])
        max_score = max(best_curves[0]['patk'])
        print(max_score)
        iterations = range(2, 40, 2)[best_curves[0]['patk'].index(max_score)]
        print('Epoch: {}'.format(iterations))


    def print_hyperparameters(self):
        
        print("Current Parameters:\n"
              "Factors: {}\n"
              "Regularization: {}\n"
              "Alpha: {}\n"
              "Iterations: {}".format(
                  self.factors, 
                  self.regularization, 
                  self.alpha, 
                  self.iterations)
              )

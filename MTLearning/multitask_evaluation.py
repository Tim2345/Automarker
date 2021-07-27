from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from scipy.stats import spearmanr

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


class MultitaskEvaluator:

    def __init__(self, y_true, y_preds, model_config, scaling):

        self.y_true = y_true.to_numpy().reshape(-1, 1)
        self.y_preds = y_preds.cpu().detach().numpy().reshape(-1,1)
        self.model_config = model_config
        self.scaling = scaling #two element tuple (mean,  std) of training data

    def scale_values(self, vals, mean, sd, direction='up'):
        '''
        function to scale values of y
        '''
        if direction == 'up':
            return vals * sd + mean
        else:
            return (vals - mean) / sd

    def apply_rules(self, score):

        if score > 28:
            score = 28
        elif score < 0:
            score = 0
        return score

    def get_reg_metrics(self, y_true=None, y_preds=None, corr=True):

        if y_true is None:
            y_true = self.y_true

        if y_preds is None:
            y_preds = self.scale_values(self.y_preds, self.scaling[0], self.scaling[1])

        mse = mean_squared_error(y_true, y_preds)
        reg_metrics_dict = {
            'samples': len(y_true),
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_preds),
            'std_true': np.std(y_true),
            'std_pred': np.std(y_preds),
            'ame': mean_absolute_error(y_true, y_preds),
            'me': np.mean(y_preds - y_true),
            'mse': mse,
            'rmse': np.sqrt(mse)
        }
        if corr==True:
            reg_metrics_dict['spearman_r'] = spearmanr(y_true, y_preds)

        return reg_metrics_dict


    def get_confusion_matrix(self):

        y_true = self.y_true
        y_preds = np.round(self.scale_values(self.y_preds, self.scaling[0], self.scaling[1]), 0)

        cf_mat = confusion_matrix(y_true, y_preds)

        heatmap = sns.heatmap(cf_mat, annot=True)
        fig = heatmap.get_figure()
        fig.savefig(self.model_config.name_or_path)



    def concat_dfs(self, y_truth, y_preds, name):
        metrics = self.get_reg_metrics(y_truth, y_preds, corr=False)
        metrics_df = pd.DataFrame(metrics, index=[0])
        name_df = pd.DataFrame({'Marks': name}, index=[0])

        return pd.concat([name_df, metrics_df], axis=1).reset_index(drop=True)

    def get_metrics_by_score(self, min_limit=0):

        y_true = self.y_true
        print(y_true)
        y_preds = np.round(self.scale_values(self.y_preds, self.scaling[0], self.scaling[1]), 0)
        print(y_preds)
        print('scaled_df')
        results_df = pd.DataFrame({'y_true': y_true.ravel(),
                                   'y_preds': y_preds.ravel()})
        print('created df')
        print(results_df.columns)
        results_df = results_df.groupby('y_true')
        print('grouped df')
        metrics_by_score_list = [self.concat_dfs(df[1]['y_true'], df[1]['y_preds'], df[0])
                                 for df in results_df
                                 if df[1].shape[0] >= min_limit
                                 ]

        metrics_by_score_df = pd.concat(metrics_by_score_list)

        return metrics_by_score_df

    def get_cross_plot(self):



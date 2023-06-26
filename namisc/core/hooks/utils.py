'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''
from ...utils import metrics

def calculate_metrics(data, metric_name):
    metric_func = getattr(metrics, metric_name)
    return metric_func(data['targets'], data['words'])
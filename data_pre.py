import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

def prepare_data(file_path):
    data = pd.read_csv(file_path)
    
    data['IPPair'] = data.apply(lambda row: tuple(sorted([row['SrcIP'], row['DstIP']])), axis=1)
    
    columns_to_use = data.columns.difference(['SrcIP', 'DstIP', 'Label', 'IPPair'])
    X = data[columns_to_use]
    y = data['Label']

    model = ExtraTreesClassifier()
    model.fit(X, y)
    feature_importances = model.feature_importances_

    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_columns = X.columns[sorted_indices]

    top_40_columns = sorted_columns[:40]
    data_to_embed = data[top_40_columns].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_to_embed)

    data_reshaped = np.expand_dims(data_scaled, axis=-1)
    
    return data, data_reshaped
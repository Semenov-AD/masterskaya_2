from flask import Flask
from flask_restful import Api, Resource

import pandas as pd
import numpy as np

import faiss
from sklearn.preprocessing import StandardScaler

from typing import Union

app = Flask(__name__)
api = Api()

dict_base = {}
for i in range(72):
    dict_base[str(i)] = 'float32'
dict_base
dict_train = dict_base.copy()
dict_train['Target'] = 'str'

data_base = pd.read_table("data/base.csv", index_col=0, sep=',', dtype=dict_base)
base_index = {k: v for k, v in enumerate(data_base.index.to_list())}

data_base.drop(['6', '21', '25', '33', '44', '59', '65', '70'], axis=1, inplace=True)

scaler = StandardScaler()
data_base_scaled = scaler.fit_transform(data_base)

data_base = pd.DataFrame(data_base_scaled, 
                         columns=data_base.columns, 
                         index=data_base.index)

dims = data_base_scaled.shape[1]
n_cells = 500
quantizer2 = faiss.IndexFlatL2(dims)
idx_l2 = faiss.IndexIVFFlat(quantizer2, dims, n_cells)

idx_l2.train(np.ascontiguousarray(data_base_scaled[:50000, :]).astype('float32'))
idx_l2.add(np.ascontiguousarray(data_base_scaled).astype('float32'))
idx_l2.nprobe = 50

class Main(Resource):
    def get(self, query: Union(str, np.array, int)):
        if isinstance(query, int):
            try:
                prod = base_index[query]
                _, idx = idx_l2.search(np.ascontiguousarray(data_base[prod]).astype('float32'), 5)
                return idx
            except ValueError:
                print('Некорректный запрос')
            
        elif isinstance(query, np.array):
            try: 
                _, idx = idx_l2.search(np.ascontiguousarray(query).astype('float32'), 5)
                return idx
            except ValueError:
                print('Некорректный запрос')
                
        elif isinstance(query, str):
            try:
                prod = data_base[query].values
                _, idx = idx_l2.search(np.ascontiguousarray(prod).astype('float32'), 5)
                return idx
            except ValueError:
                print('Некорректный запрос')
    
api.add_resource(Main, '/api/main/query')
api.init_app(app)


if __name__ == '__main__':
    app.run(debug=True, port=3000, host='127.0.0.1')
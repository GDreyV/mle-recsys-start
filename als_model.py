import implicit.cpu.als
import implicit.gpu.als
from implicit.als import AlternatingLeastSquares
import pandas as pd
import scipy
import sklearn.preprocessing
import numpy as np
import sys
import os
import joblib
from typing import TypeAlias, cast
import time

ALSModel: TypeAlias = implicit.gpu.als.AlternatingLeastSquares | implicit.cpu.als.AlternatingLeastSquares

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class ALSPredictor:
    def __init__(self, items: pd.DataFrame, events: pd.DataFrame, events_train: pd.DataFrame, events_test: pd.DataFrame):
        self._items = items
        self._events = events
        self._events_train = events_train
        self._events_test = events_test
        self._encode_data()
        self._model = None

    def fit(self) -> int:
        """
        Инициализация модели ALS.
        """
        self._encode_data()

        if self._model is None:
            self._model = self._train_or_load()

        return self._events_train["item_id_enc"].max()

    def _encode_data(self) -> None:
        start = time.time()
        print("Encoding data...")
        self._encode_user()
        print(f"User encoding took {time.time() - start:.2f} seconds")

        start = time.time()
        self._encode_item()
        print(f"Item encoding took {time.time() - start:.2f} seconds")

        start = time.time()
        self._construct_user_item_matrix()
        print(f"User-item matrix construction took {time.time() - start:.2f} seconds")

    def _encode_user(self) -> None:
        # перекодируем идентификаторы пользователей:
        # из имеющихся в последовательность 0, 1, 2, ...
        self._user_encoder = sklearn.preprocessing.LabelEncoder()
        self._user_encoder.fit(self._events["user_id"])
        self._events_train["user_id_enc"] = self._user_encoder.transform(self._events_train["user_id"])
        self._events_test["user_id_enc"] = self._user_encoder.transform(self._events_test["user_id"])

    def _encode_item(self) -> None:
        # перекодируем идентификаторы объектов: 
        # из имеющихся в последовательность 0, 1, 2, ...
        self._item_encoder = sklearn.preprocessing.LabelEncoder()
        self._item_encoder.fit(self._items["item_id"])
        self._items["item_id_enc"] = self._item_encoder.transform(self._items["item_id"])
        self._events_train["item_id_enc"] = self._item_encoder.transform(self._events_train["item_id"])
        self._events_test["item_id_enc"] = self._item_encoder.transform(self._events_test["item_id"])

    def _construct_user_item_matrix(self) -> None:
        self._user_item_matrix = scipy.sparse.csr_matrix((
            self._events_test["rating"],
            (self._events_test['user_id_enc'], self._events_test['item_id_enc'])),
            dtype=np.int8)
    
    def _train_or_load(self) -> ALSModel:
        path = os.path.abspath(f"{_CURRENT_DIR}/data/models/als_model.pkl")
        if os.path.exists(path):
            print(f"Loading ALS model from {path}")
            return cast(ALSModel, joblib.load(path))

        print(f"Training ALS model and saving to {path}")
        als_model = self._train()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(als_model, path)
        return als_model

    def _train(self) -> ALSModel:
        events_train = self._events_train
        matrix_size = (events_train["user_id_enc"].max() + 1) \
            * (events_train["item_id_enc"].max() + 1) / 1024**3  # in GB
        print(f"Expected matrix size is {matrix_size:.2f} GB")

        # создаём sparse-матрицу формата CSR
        user_item_matrix_train = scipy.sparse.csr_matrix((
            events_train["rating"],
            (events_train['user_id_enc'], events_train['item_id_enc'])),
            dtype=np.int8)

        matrix_size = sum([sys.getsizeof(i) for i in user_item_matrix_train.data])/1024**3
        print(f"Real matrix size {matrix_size:.2f} GB")

        als_model: ALSModel = AlternatingLeastSquares(factors=50, iterations=50, regularization=0.05, random_state=0)
        als_model.fit(user_item_matrix_train)
        return als_model

    def recommend(self, user_id, include_seen=True, n=5):
        """
        Возвращает отранжированные рекомендации для заданного пользователя
        """
        assert self._model is not None, "Model is not initialized. Call init() first."
        assert self._user_encoder is not None, "User encoder is not initialized."
        assert self._item_encoder is not None, "Item encoder is not initialized."

        user_id_enc = self._user_encoder.transform([user_id])[0]
        recommendations = self._model.recommend(
            user_id_enc,
            self._user_item_matrix[user_id_enc],
            filter_already_liked_items=not include_seen,
            N=n)
        recommendations = pd.DataFrame({"item_id_enc": recommendations[0], "score": recommendations[1]})
        recommendations["item_id"] = self._item_encoder.inverse_transform(recommendations["item_id_enc"])
        
        return recommendations

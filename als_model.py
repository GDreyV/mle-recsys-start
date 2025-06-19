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
import sklearn.metrics

ALSModel: TypeAlias = (
    implicit.gpu.als.AlternatingLeastSquares | implicit.cpu.als.AlternatingLeastSquares
)

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_RECOMMENDATIONS_PATH = os.path.abspath(
    f"{_CURRENT_DIR}/data/als_recommendations.parquet"
)


class ALSPredictor:
    _als_recommendations: pd.DataFrame | None = None

    def __init__(
        self,
        items: pd.DataFrame,
        events: pd.DataFrame,
        events_train: pd.DataFrame,
        events_test: pd.DataFrame,
    ):
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
        self._events_train["user_id_enc"] = self._user_encoder.transform(
            self._events_train["user_id"]
        )
        self._events_test["user_id_enc"] = self._user_encoder.transform(
            self._events_test["user_id"]
        )

    def _encode_item(self) -> None:
        # перекодируем идентификаторы объектов:
        # из имеющихся в последовательность 0, 1, 2, ...
        self._item_encoder = sklearn.preprocessing.LabelEncoder()
        self._item_encoder.fit(self._items["item_id"])
        self._items["item_id_enc"] = self._item_encoder.transform(
            self._items["item_id"]
        )
        self._events_train["item_id_enc"] = self._item_encoder.transform(
            self._events_train["item_id"]
        )
        self._events_test["item_id_enc"] = self._item_encoder.transform(
            self._events_test["item_id"]
        )

    def _construct_user_item_matrix(self) -> None:
        # создаём sparse-матрицу формата CSR
        self._user_item_matrix = scipy.sparse.csr_matrix(
            (
                self._events_test["rating"],
                (self._events_test["user_id_enc"], self._events_test["item_id_enc"]),
            ),
            dtype=np.int8,
        )

        self._user_item_matrix_train = scipy.sparse.csr_matrix(
            (
                self._events_train["rating"],
                (self._events_train["user_id_enc"], self._events_train["item_id_enc"]),
            ),
            dtype=np.int8,
        )

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
        matrix_size = (
            (events_train["user_id_enc"].max() + 1)
            * (events_train["item_id_enc"].max() + 1)
            / 1024**3
        )  # in GB
        print(f"Expected matrix size is {matrix_size:.2f} GB")

        matrix_size = (
            sum([sys.getsizeof(i) for i in self._user_item_matrix_train.data]) / 1024**3
        )
        print(f"Real matrix size {matrix_size:.2f} GB")

        als_model: ALSModel = AlternatingLeastSquares(
            factors=50, iterations=50, regularization=0.05, random_state=0
        )
        als_model.fit(self._user_item_matrix_train)
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
            N=n,
        )
        recommendations = pd.DataFrame(
            {"item_id_enc": recommendations[0], "score": recommendations[1]}
        )
        recommendations["item_id"] = self._item_encoder.inverse_transform(
            recommendations["item_id_enc"]
        )

        return recommendations

    def _predict_all(self) -> pd.DataFrame:
        assert self._model is not None, "Model is not initialized. Call fit() first."

        # получаем список всех возможных user_id (перекодированных)
        user_ids_encoded = range(len(self._user_encoder.classes_))

        # получаем рекомендации для всех пользователей
        als_recommendations = self._model.recommend(
            user_ids_encoded,
            self._user_item_matrix_train[user_ids_encoded],
            filter_already_liked_items=False,
            N=100,
        )

        # преобразуем полученные рекомендации в табличный формат
        item_ids_enc = als_recommendations[0]
        als_scores = als_recommendations[1]

        als_recommendations = pd.DataFrame(
            {
                "user_id_enc": user_ids_encoded,
                "item_id_enc": item_ids_enc.tolist(),
                "score": als_scores.tolist(),
            }
        )
        als_recommendations = als_recommendations.explode(
            ["item_id_enc", "score"], ignore_index=True
        )

        # приводим типы данных
        als_recommendations["item_id_enc"] = als_recommendations["item_id_enc"].astype(
            "int"
        )
        als_recommendations["score"] = als_recommendations["score"].astype("float")

        # получаем изначальные идентификаторы
        als_recommendations["user_id"] = self._user_encoder.inverse_transform(
            als_recommendations["user_id_enc"]
        )
        als_recommendations["item_id"] = self._item_encoder.inverse_transform(
            als_recommendations["item_id_enc"]
        )
        als_recommendations = als_recommendations.drop(
            columns=["user_id_enc", "item_id_enc"]
        )

        als_recommendations = als_recommendations[["user_id", "item_id", "score"]]
        als_recommendations.to_parquet(_RECOMMENDATIONS_PATH)
        print(f"ALS recommendations saved to {_RECOMMENDATIONS_PATH}")
        return als_recommendations

    def get_all_recommendations(self) -> pd.DataFrame:
        """
        Возвращает все рекомендации, сохранённые в файле.
        """
        if self._als_recommendations is not None:
            return self._als_recommendations

        if os.path.exists(_RECOMMENDATIONS_PATH):
            print(f"Loading ALS recommendations from {_RECOMMENDATIONS_PATH}")
            self._als_recommendations = pd.read_parquet(_RECOMMENDATIONS_PATH)
        else:
            print(
                f"No ALS recommendations found at {_RECOMMENDATIONS_PATH}, computing..."
            )
            self._als_recommendations = self._predict_all()

        return self._als_recommendations

    def _compute_ndcg(self, rating: pd.Series, score: pd.Series, k):
        """
        подсчёт ndcg
        rating: истинные оценки
        score: оценки модели
        k: количество айтемов (по убыванию score) для оценки, остальные - отбрасываются
        """
        # если кол-во объектов меньше 2, то NDCG - не определена
        if len(rating) < 2:
            return np.nan

        ndcg = sklearn.metrics.ndcg_score(
            np.asarray([rating.to_numpy()]), np.asarray([score.to_numpy()]), k=k
        )

        return ndcg

    def compute_rating(self) -> tuple[float, float]:
        """
        Returns the average NDCG@5 score and the percentage of users covered
        by the NDCG computation in the test set.
        """
        als_recommendations = self.get_all_recommendations()
        als_recommendations = als_recommendations.merge(
            self._events_test[["user_id", "item_id", "rating"]].rename(
                columns={"rating": "rating_test"}
            ),
            on=["user_id", "item_id"],
            how="left",
        )

        rating_test_idx = ~als_recommendations["rating_test"].isnull()
        ndcg_at_5_scores = (
            als_recommendations[rating_test_idx]
            .groupby("user_id")
            .apply(lambda x: self._compute_ndcg(x["rating_test"], x["score"], k=5))
        )

        # Number of users with at least one rating_test value (i.e., NDCG was computed)
        users_with_ndcg = als_recommendations.loc[rating_test_idx, "user_id"].nunique()
        # Total number of users in the test set
        total_test_users = self._events_test["user_id"].nunique()
        # Compute percentage
        percentage_covered = 100 * users_with_ndcg / total_test_users

        return ndcg_at_5_scores.mean(), percentage_covered

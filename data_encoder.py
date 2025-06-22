import pandas as pd
import sklearn.preprocessing
import time


class DataEncoder:
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

    @property
    def user_encoder(self) -> sklearn.preprocessing.LabelEncoder:
        """
        Возвращает перекодировщик пользователей.
        """
        return self._user_encoder

    @property
    def item_encoder(self) -> sklearn.preprocessing.LabelEncoder:
        """
        Возвращает перекодировщик объектов.
        """
        return self._item_encoder

    @property
    def items(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с перекодированными объектами.
        """
        return self._items

    @property
    def train(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с перекодированными событиями для обучающей выборки.
        """
        return self._events_train

    @property
    def test(self) -> pd.DataFrame:
        """
        Возвращает DataFrame с перекодированными событиями для тестовой выборки.
        """
        return self._events_test

    def fit(self) -> None:
        """
        Выполняет перекодировку данных.
        """
        self._encode_data()

    def _encode_data(self) -> None:
        start = time.time()
        print("Encoding data...")
        self._encode_user()
        print(f"User encoding took {time.time() - start:.2f} seconds")

        start = time.time()
        self._encode_item()
        print(f"Item encoding took {time.time() - start:.2f} seconds")

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

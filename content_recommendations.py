import pandas as pd
import scipy
import sklearn.preprocessing


def get_genres(items: pd.DataFrame) -> pd.DataFrame:
    """
    извлекает список жанров по всем книгам,
    подсчитывает долю голосов по каждому их них
    """

    genres_counter: dict[str, int] = {}

    for (
        k,
        v,
    ) in items.iterrows():
        genre_and_votes = v["genre_and_votes"]
        if genre_and_votes is None or not isinstance(genre_and_votes, dict):
            continue
        for genre, votes in genre_and_votes.items():
            # увеличиваем счётчик жанров
            try:
                genres_counter[genre] += votes
            except KeyError:
                genres_counter[genre] = 0

    genres = pd.Series(genres_counter, name="votes")
    genres = genres.to_frame()
    genres = genres.reset_index().rename(columns={"index": "name"})
    genres.index.name = "genre_id"

    return genres


def get_item2genre_matrix(genres: pd.DataFrame, items: pd.DataFrame):

    genre_names_to_id = genres.reset_index().set_index("name")["genre_id"].to_dict()

    # list to build CSR matrix
    genres_csr_data = []
    genres_csr_row_idx = []
    genres_csr_col_idx = []

    for item_idx, (k, v) in enumerate(items.iterrows()):
        if v["genre_and_votes"] is None:
            continue
        for genre_name, votes in v["genre_and_votes"].items():
            genre_idx = genre_names_to_id[genre_name]
            genres_csr_data.append(int(votes))
            genres_csr_row_idx.append(item_idx)
            genres_csr_col_idx.append(genre_idx)

    genres_csr = scipy.sparse.csr_matrix(
        (genres_csr_data, (genres_csr_row_idx, genres_csr_col_idx)),
        shape=(len(items), len(genres)),
    )
    # нормализуем, чтобы сумма оценок принадлежности к жанру была равна 1
    genres_csr = sklearn.preprocessing.normalize(genres_csr, norm="l1", axis=1)

    return genres_csr

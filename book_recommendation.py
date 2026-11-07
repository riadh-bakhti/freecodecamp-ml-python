import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

df_books = pd.read_csv(
    "data/BX-Books.csv",
    encoding="ISO-8859-1", sep=";", header=0,
    names=["isbn", "title", "author"],
    usecols=["isbn", "title", "author"],
    dtype={"isbn": "str", "title": "str", "author": "str"})

df_ratings = pd.read_csv(
    "data/BX-Book-Ratings.csv",
    encoding="ISO-8859-1", sep=";", header=0,
    names=["user", "isbn", "rating"],
    usecols=["user", "isbn", "rating"],
    dtype={"user": "int32", "isbn": "str", "rating": "float32"})

# Remove users with fewer than 200 ratings and books with fewer than 100
# ratings. The counts are taken from the original dataset.
user_counts = df_ratings["user"].value_counts()
book_counts = df_ratings["isbn"].value_counts()
df_ratings = df_ratings[df_ratings["user"].isin(user_counts[user_counts >= 200].index)]
df_ratings = df_ratings[df_ratings["isbn"].isin(book_counts[book_counts >= 100].index)]

df = df_ratings.merge(df_books, on="isbn", how="inner")
df = df.drop_duplicates(["user", "title"])

book_user_matrix = df.pivot(index="title", columns="user", values="rating").fillna(0)
matrix = csr_matrix(book_user_matrix.values)

model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
model.fit(matrix)


def get_recommends(book=""):
    distances, indices = model.kneighbors(
        book_user_matrix.loc[book].values.reshape(1, -1), n_neighbors=6)
    # Return 5 books, farthest first (as required by the fCC test).
    recommended_books = [
        [book_user_matrix.index[indices[0][i]], float(distances[0][i])]
        for i in range(5, 0, -1)
    ]
    return [book, recommended_books]


if __name__ == "__main__":
    books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
    for title, dist in books[1]:
        print(f"{dist:.4f}  {title}")

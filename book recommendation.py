
# item based filtering, using KNN on books (ISBN)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

books_filename = 'data/BX-Books.csv'
ratings_filename = 'data/BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# remove from the dataset users with less than 200 ratings and books with less than 100 ratings.
userCounts = df_ratings['user'].value_counts()
isbnCounts = df_ratings['isbn'].value_counts()
#remove all users with less than 200 reviews & remove all books with less than 100 ratings
df_ratings = df_ratings[~df_ratings['user'].isin(userCounts[userCounts < 200].index) & ~df_ratings['isbn'].isin(isbnCounts[isbnCounts < 100].index)]

# merge ratings with books into one dataframe
dataframe = pd.merge(df_ratings, df_books, on='isbn')

# create a separate dataframe with the total number of ratings on each book
combine_book_rating = dataframe.dropna(axis=0, subset=['title'])
book_rating_count = (combine_book_rating.groupby(by=['title'])['rating'].count().
                     reset_index().rename(columns={'rating': 'totalRatingCount'})
                     [['title', 'totalRatingCount']])

# merge the rating count dataframe with 'dataframe'
mergedDataframe = combine_book_rating.merge(book_rating_count, left_on='title', right_on='title',
                                            how='left')
# create pivot table
pivot_table = mergedDataframe.pivot_table(index='isbn', columns='user', values='rating').fillna(0)

# convert to sparse matrix
book_matrix = csr_matrix(pivot_table.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(book_matrix)
# function to return recommended books - this will be tested
def get_recommends(book = ""):
    isbn = mergedDataframe.query(f'title=="{book}"')['isbn'].iloc[0]
    recommended_books = []
    recommended_books.append(book)
    distances, indices = model_knn.kneighbors(
        pivot_table.loc[isbn, :].values.reshape(1, -1),
        n_neighbors=6)
    books = []
    for i in range(1, len(distances.flatten())):
        books.append([mergedDataframe.query(f'isbn=="{pivot_table.index[indices.flatten()[i]]}"')['title'].iloc[0], float(distances.flatten()[i])])
    books = sorted(books, key=lambda x: x[1], reverse=True)
    recommended_books.append(books)
    return recommended_books

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()





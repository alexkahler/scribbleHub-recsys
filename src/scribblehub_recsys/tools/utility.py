from ast import literal_eval
import logging
import pickle

from scipy import sparse as sp
import pandas as pd
import numpy as np
import zstandard


def load_data(dataset):
    """Load data from file.

    Returns:
        Pandas Dataframe: Novels or Reading-Lists depending on choice.
    """
    
    if dataset == "reading-lists":
        dtypes={"web-scraper-order":"string", 
                "user_href":"string",
                "user_id":"Int32", 
                "username":"string", 
                "novel_id":"Int32",
                "novel":"string", 
                "novel_href":"string",
                "key":"string"}
        
        file = "./datasets/reading-lists.txt"
        
        df = pd.read_csv(file, 
                         encoding='utf-8',
                         sep='\t', 
                         decimal=',',
                         dtype=dtypes,
                         na_values={"novel_id":"np.nan"}
                         ).drop_duplicates().reset_index()
        
        df.dropna(subset=['user_id']).dropna(subset=['novel_id'])
        
    elif dataset == "novels":
        file = "./models/novels.zst"
        df = decompress_unpickle(file)
    
    else:
        return None
        
    return df


def filter_short_novels(novels_df):
    """Drop all novels with less than 5 chapters.
    
    """
    
    return novels_df[novels_df.chapters > 5].reset_index(drop=True)

# Credit to datacamp.com/tutorial/recommender-systems-python
def weighted_rating(df, mean_rating, quantile_votes):
    """Calculate weighted rating based on IMDB's rating formula.
    
    Args:
        param1: Pandas DataFrame.
        param2: Mean rating of the votes.
        param3: The number of votes in the 90th quantile.
    
    Returns:
        int: The weighted rating.
    """
    
    vote_count = df['rating_votes']
    rating = df['rating']
    return ((vote_count / (vote_count + quantile_votes) * rating) +
            (quantile_votes / (quantile_votes + vote_count) * mean_rating)) 

def parse_tags(novels_df, features):
    """Convert the JSON formatted tags, genres, fandom_tags into a list of strings.
    
    """
    
    if features is None:
        features = ['tags', 'genres', 'fandom_tags']
    
    for feature in features:
        novels_df[feature] = novels_df[feature].apply(literal_eval)
        # novels_df[feature] = novels_df[feature].apply(convert_json_tags, args=(feature, ))

    return novels_df


def convert_json_tags(json, key):
    """Convert JSON object into a list of strings

    Args:
        json (list): List of JSON objects.
        key (string): Name of the key which has the value.

    Returns:
        List: List of Strings.
    """
    
    if isinstance(json, list):
        tags = [i[key] for i in json]
        return tags
    return []


def filter_library_size(df, user_id_min, novel_id_min):
    """Filter the reading list dataframe based on the minimum number of novels added to 
    the reading list and users who have not added the minimum amount of books.

    Args:
        df (Pandas Dataframe): DataFrame of reading list.
        user_id_min (_type_): User's minimum novel amount.
        novel_id_min (_type_): Novel's minimum number of users.

    Returns:
        _type_: _description_
    """
    n_users = df.user_id.unique().shape[0]
    n_items = df.novel_id.unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    logging.debug('Starting reading list size info:')
    logging.debug('Number of users: {}'.format(n_users))
    logging.debug('Number of novels: {}'.format(n_items))
    logging.debug('Sparsity: {:4.3f}%'.format(sparsity))
    
    done = False
    while not done:
        starting_shape = df.shape[0]
        novel_id_counts = df.groupby('user_id').novel_id.count()
        df = df[~df.user_id.isin(novel_id_counts[novel_id_counts < novel_id_min].index.tolist())]
        user_id_counts = df.groupby('novel_id').user_id.count()
        df = df[~df.novel_id.isin(user_id_counts[user_id_counts < user_id_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True
    
    assert(df.groupby('user_id').novel_id.count().min() >= novel_id_min)
    assert(df.groupby('novel_id').user_id.count().min() >= user_id_min)
    
    n_users = df.user_id.unique().shape[0]
    n_items = df.novel_id.unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_users*n_items) * 100
    
    logging.debug('Ending reading list size info:')
    logging.debug('Number of users: {}'.format(n_users))
    logging.debug('Number of novels: {}'.format(n_items))
    logging.debug('Sparsity: {:4.3f}%'.format(sparsity))
    
    return df


def process_recommendation(
    user_id, 
    recommendations, 
    scores,
    data,
    user_to_index,
    index_to_novel,
    reading_lists):
    
    results = {}
    for i in range(0, len(recommendations)):
        results[index_to_novel[recommendations[i]]] = [reading_lists.loc[reading_lists["novel_id"]==index_to_novel[recommendations[i]], "novel"].iloc[0], scores[i], np.in1d(recommendations[i], data[user_to_index[user_id]].indices)]

    return pd.DataFrame.from_dict(results, orient='index', columns=["novel", "score", "viewed"])


def process_simillar_items(
    user_id,
    recommendations,
    scores,
    data,
    reading_lists,
    user_to_index,
    index_to_novel):
    
    results = {}
    for i in range(0, len(recommendations)):
        results[index_to_novel[recommendations[i]]] = [reading_lists.loc[reading_lists["novel_id"]==index_to_novel[recommendations[i]], "novel"].iloc[0], scores[i], np.in1d(recommendations[i], data[user_to_index[user_id]].indices)]

    # display the results using pandas for nicer formatting
    return pd.DataFrame.from_dict(results, orient='index', columns=["novel", "score", "viewed"])


def create_index_mappings(df, column):
    
    idx_user = pd.Series(np.sort(np.unique(df[column])))
    user_idx = pd.Series(data=idx_user.index, index=idx_user.values)
    return idx_user, user_idx


def create_csr_matrix(df, user_idx, novel_idx):
    
    rows = df["user_id"].map(user_idx)
    cols = df["novel_id"].map(novel_idx)
    values = np.ones(rows.shape[0])
    #Format the data into CSR Matrix for the model.
    # Create a user-item matrix from the dataframe
    #data = sp.coo_matrix((np.ones(len(reading_lists.user_id)), (reading_lists['user_id'], reading_lists['novel_id']))).tocsr()
    #data = [tuple(x) for x in reading_lists[['user_id', 'novel_id']].to_records(index=False)]
    return sp.csr_matrix((values, (rows, cols)),shape=(df.user_id.unique().shape[0], df.novel_id.unique().shape[0]))


def compress_pickle(obj, file_path):
    binary_data = pickle.dumps(obj)
    compressed_data = zstandard.compress(binary_data)
    with open(file_path, "wb") as f:
        f.write(compressed_data)


def decompress_unpickle(file_path):
    with open(file_path, "rb") as f:
        compressed_data = f.read()
    binary_data = zstandard.decompress(compressed_data)
    obj = pickle.loads(binary_data)
    return obj



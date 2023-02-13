import os

from _engines import cf_engine, cb_engine
from tools import utility, constants

path = r"C:\\Users\\akaehler\\OneDrive - GN Store Nord\Documents\ScribbleHub\ScribbleHub RecSys\\models\\"

#TODO Clean up code structure to allow for better maintanence and export of models and dataframes for use with Streamlit app.
#TODO: Create CLI interface which uses both CB and CF recommender engines.
def cf_recommendation():
    load_model = True
    save_model = False
    load_mappings = True
    save_mappings = False
    train_model = False
    do_grid_search = False
    do_recommendation = True
    
    reading_lists = utility.load_data("reading-lists")
        
    #Cut off any users with less than 5 books in library, or any books with less than 5 readers.
    reading_lists = utility.filter_library_size(reading_lists, 5, 5).reset_index()
    
    if load_mappings:
        mappings = utility.decompress_unpickle("./models/mappings.zst")
        data = mappings.get('data')
        index_to_novel = mappings.get('index_to_novel')
        novel_to_index = mappings.get('novel_to_index')
        user_to_index = mappings.get('user_to_index')
        
    else:
        _, user_to_index = utility.create_index_mappings(reading_lists, "user_id")
        index_to_novel, novel_to_index = utility.create_index_mappings(reading_lists, "novel_id")
        data = utility.create_csr_matrix(reading_lists, user_to_index, novel_to_index)    

    cf_recommender = cf_engine.CFRecommender(alpha=10, regularization=10, factors=80, iterations=16)
    
    if load_model:
        cf_recommender = cf_recommender.load(os.path.join(path, "implicit_model.npz"))
        
    else:
        if train_model:
            cf_recommender.fit(data)
        
        elif do_grid_search:
            cf_recommender.grid_search(data)
            
        if save_model:
            cf_recommender.save(os.path.join(path, "implicit_model"))
    
    if save_mappings: # TODO: Save all mappings, model and csr matrix in one dict. Save with pickle and zstd compression.
        utility.compress_pickle(
            {'data': data, 
             'user_to_index': user_to_index,
             'novel_to_index': novel_to_index,
             'index_to_novel': index_to_novel},
            "./models/mappings.zst")
            
    if do_recommendation:
        user_id, novel_id, top_n = 17062, 124643, 10
        # Get recommendations for a specific user    
        recommendations, scores = cf_recommender.recommend(
            user_to_index[user_id], 
            data[user_to_index[user_id]],
            N=top_n, 
            filter_already_liked_items=True)
        
        print("Looking up recommendations for {}:".format(
            reading_lists.loc[reading_lists["user_id"]==user_id, "username"].iloc[0]))
        
        print(utility.process_recommendation(
            user_id=user_id, 
            recommendations=recommendations, 
            scores=scores, 
            data=data, 
            user_to_index=user_to_index, 
            index_to_novel=index_to_novel, 
            reading_lists=reading_lists))
        
        # Get similar item
        recommendations, scores = cf_recommender.similar_items(
            novel_to_index[novel_id], 
            N=top_n)
        
        print("Getting novels similar to {}:".format(
            reading_lists.loc[reading_lists["novel_id"]==novel_id, "novel"].iloc[0]))
        
        #TODO: Refactor so indexes are in class.
        print(utility.process_simillar_items(
            user_id=user_id, 
            recommendations=recommendations, 
            scores=scores, 
            data=data, 
            reading_lists=reading_lists, 
            user_to_index=user_to_index, 
            index_to_novel=index_to_novel))
        
        cf_recommender.print_hyperparameters()

def cb_recommendation(
    novel_id: int = 681902, 
    number: int = 10, 
    do_recommendation: bool = True, 
    do_ranking: bool = True):
    
    novels_df = utility.load_data('novels')
    novels_df = utility.filter_short_novels(novels_df)
    # print(novels_df['novel_id'].iloc[:10])
    cb_recommender = cb_engine.CBRecommender(novels_df)
    similarity_matrix, indices = cb_recommender.fit(novels_df)
    
    if do_recommendation:
        recommended_novels = cb_recommender.recommend(
            novel_id, 
            number, 
            similarity_matrix, 
            indices, 
            novels_df)

        print(recommended_novels)
    
    if do_ranking:
        print(cb_recommender.ranking(novels_df))


import pandas as pd
import numpy as np
def export_novels_for_streamlit():
    dtypes = {
        'author': str,
        'author_href': str,
        'author_id': np.int32,
        'average_views': np.int32,
        'average_words': np.int32,
        'chapters': np.int16,
        'chapters_per_week': np.int8,
        'fandom_tags': list,
        'favorites': np.int32,
        'genres': list,
        'image_src': str,
        'last_update': str,
        'novel_href': str,
        'novel_id': np.int32,
        'pages': np.int32,
        'rating': np.float32,
        'rating_votes': np.int16,
        'readers': np.int16,
        'reviews_count': np.int16,
        'status': str,
        'synopsis': list,
        'tags': list,
        'title': str,
        'total_views_all': np.int32,
        'total_views_chapters': np.int32,
        'word_count': np.int32,
    }
    
    novels_df = pd.read_json(
        path_or_buf="./datasets/novels-2023-02-09T20-34-48.jsonl", 
        lines=True,
        dtype=dtypes)
    
    print("Loaded novel: ")
    print(novels_df.info())
    print(novels_df.describe())
    print(novels_df.sample(10))
    novels_df.dropna(
        subset=['author_id', 'novel_id'], 
        inplace=True)
    
    novels_df.drop_duplicates(subset='novel_id', inplace=True)
    
    mean_rating = novels_df['rating'].mean()
    quantile_votes = novels_df['rating_votes'].quantile(0.90)
    
    
    novels_df['weighted_rating'] = novels_df.apply(
        utility.weighted_rating, 
        args=(mean_rating, quantile_votes,),
        axis=1)
    
    novels_df['status'] = novels_df['status'].str.strip().astype('category')
    
    novels_df.drop(
        ['rating', 'author_href', 'novel_href'],
        axis=1,
        inplace=True)
    novels_df = novels_df.drop_duplicates('novel_id').reset_index()
    
    genres = constants.GENRES
    tags = constants.TAGS
    novels_df['genres'] = novels_df['genres'].apply(lambda x: [i for i in x if i in genres])
    novels_df['tags'] = novels_df['tags'].apply(lambda x: [] if x is None else x)
    novels_df['tags'] = novels_df['tags'].apply(lambda x: [i for i in x if i in tags])
    
    print("End result:")
    print(novels_df.info())
    print(novels_df.describe())
    print(novels_df.sample(10))
    cont = input('Do you wish to continue with export? (Y/N) ')
    
    if cont.lower() == 'y':
        utility.compress_pickle(novels_df, './models/novels.zst')
        print('Output successful.')
    
    else:
        print('Cancelled output.')
    #TOOD Drop novels without chapters, if loading times are too long in streamlit.
    

if __name__ == "__main__":
    n_export = True
    cb_rec = False
    cf_rec = False
    if n_export:
        export_novels_for_streamlit()
    if cb_rec:
        cb_recommendation()
    if cf_rec:
        cf_recommendation()
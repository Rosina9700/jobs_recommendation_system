import pandas as pd
import numpy as np
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

def get_feature_data():
    df = pd.read_csv('../data/all_with_features.csv')
    df['campaign_start_date'] = pd.to_datetime(df['campaign_start_date'])
    df = df.drop_duplicates()
    df = df.drop_duplicates(['description'])
    return df.reset_index()

def get_original_data():
    df = pd.read_csv('../data/all_jobs.csv')
    df['campaign_start_date'] = pd.to_datetime(df['campaign_start_date'])
    df['campaign_end_date'] = pd.to_datetime(df['campaign_end_date'])
    return df

def generate_recommendations(idx, train, df_orig):
    '''Function to generate a recommendation from a given index
    Finds the top valid similar jobs and prints them.
    INPUTS
    ---------------------------
    idx: Integer
    train: Pandas DataFrame
    df_orig: Pandas DataFrame
    '''
    # find original index
    original_index = train.iloc[idx,:]['orig_idx']
    # print 'original_index: {}'.format(original_index)
    print 'ORIGINAL:'
    print train.iloc[idx,:]['title']
    print df_orig.iloc[original_index,:][['title','job_level','country','campaign_start_date','campaign_end_date','description','employer_name']]
    # find start date for that add as don't want to recommend ads which have expired before this one started
    start_date = df_orig.iloc[original_index,:]['campaign_start_date']
    print '\n*************************'
    print 'Top similar jobs .......'
    top_5_orig_index = train.iloc[idx,:]['top_recs']
    temp = df_orig.iloc[top_5_orig_index,:]
    print temp[temp['campaign_end_date']>start_date][['title','job_level','country','campaign_start_date','campaign_end_date','description','employer_name']]
    print '----------------------------------------------\n'
    pass

def vectorize_text_feature(df, feature, num_vec_features):
    """ Function to vectorize a text description so that it can be used in the recommendation system.
    Removes punctuation, selects alpha numeric characters only and stems words.
    tf_idf vectoriser used with maximum and minimum document frequency of 0.5 and 0.05 respectively to capture important common words.
    number of vector features defined by user.
    INPUTS:
    --------------------------
    df: Pandas DataFrame
    feature: string
    num_vec_features: Integer
    RETURNS:
    --------------------------
    vectorized: Pandas DataFrame
    """
    # initialize NLP tools
    punctuation = set(string.punctuation)
    stemmer = SnowballStemmer('english')
    tf_vec= TfidfVectorizer(max_df=0.5, min_df=0.005, stop_words='english',ngram_range=(1,3),max_features=num_vec_features)

    # prepare text feature for vectorization
    df[feature +'_no_punc'] = df[feature].map(lambda x: ''.join(ch for ch in x if x not in punctuation))
    df[feature + '_stemmed'] = df[feature].map(lambda x: ' '.join([stemmer.stem(y) for y in x.decode('utf-8').split(' ') if y.isalpha()]))

    # fit and transform feature
    tf_vec = tf_vec.fit(df[feature+'_stemmed'].values)
    vectorized = tf_vec.transform(df[feature+'_stemmed'].values).toarray()

    # convert to Pandas Dataframe with appropriate column names
    col = [feature+ '_v_'+str(i) for i in xrange(num_vec_features)]
    vectorized = pd.DataFrame(data=vectorized, columns=col)

    # drop intermediary columns from original dataframe
    df.drop([feature +'_no_punc',feature +'_stemmed'],axis=1,inplace=True)
    return vectorized

if __name__=='__main__':
    # import data
    train = get_feature_data()
    df_orig = get_original_data()

    # Whether to vectorize original data or not
    vectorise_text = True
    if vectorise_text == True:
        vectorized_desciption = vectorize_text_feature(train, 'description', 15)
        vectorized_title = vectorize_text_feature(train, 'title',5)
        train2 =  train.drop(['description','orig_idx','title','campaign_start_date'],axis=1)
        train2 = pd.concat([train2, vectorized_desciption, vectorized_title], axis=1)
    else:
        train2 =  train.drop(['description','orig_idx','title','campaign_start_date'],axis=1)

    # standardize data
    scaler = StandardScaler()
    scaler = scaler.fit(train2)
    train_scaled = scaler.transform(train2)

    # use principle component analysis to reduce dimensions
    pca = PCA(n_components=20)
    pca = pca.fit(train_scaled)
    train_reduced = pca.transform(train_scaled)

    # calculate matrix of cosine similarity
    distances = cosine_similarity(train_reduced).T

    # find top 5 closest ads
    top_5_orig = []
    top_distances = []
    for i in xrange(len(train_reduced)):
        # the closest will be itself so take top 6-1 closest
        top = distances[i,:].argsort()[-6:-1][::-1]
        dists  = distances[i,top]
        top_orig = train.iloc[top,:]['orig_idx'].values
        top_5_orig.append(top_orig)
        top_distances.append(dists)

    # add top recommendations and respective distances to the training DataFrame
    train['top_recs'] = pd.Series(top_5_orig)
    train['top_dists'] = pd.Series(top_distances)

    # print to csv so that recommendations can be accessed later using the generate_recommendations function
    train.to_csv('../data/all_with_recs.csv',header=True,index=False,encoding='utf-8')

    # generate a recommendation
    generate_recommendations(150, train, df_orig)

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
    df = pd.read_csv('data/all_with_features.csv')
    cols_to_keep = [u'title',u'Unnamed: 0', u'description', u'number_of_positions', u'required_work_experience_months',
       u'campaign_start_date_year', u'campaign_start_date_month', u'campaign_start_date_day',
       u'campaign_start_date_dow', u'campaign_duration', u'desc_length',
       u'country_code_CI', u'country_code_DZ', u'country_code_KE',
       u'country_code_MU', u'country_code_MW', u'country_code_NG',
       u'country_code_PK', u'country_code_RW', u'country_code_SN',
       u'country_code_TZ', u'country_code_UG', u'country_code_ZA',
       u'country_code_nan', u'job_level_Entry-Level', u'job_level_Mid-Level',
       u'job_level_Senior', u'job_level_nan', u'location_Nairobi',
       u'location_Uganda', u'location_Tharaka Nithi', u'location_Kajiado',
       u'location_Kisii', u'location_Mombasa', u'location_Daadab',
       u'location_Kisumu', u'location_other', u'employer_Chuka University',
       u'employer_Summit Recruitment &Training',
       u'employer_National Drug Authority (Uganda)',
       u'employer_Export Promotion Council',
       u'employer_Tharaka Nithi County Government',
       u'employer_Umma University ', u'employer_Microsoft 4Afrika Skills',
       u'employer_CARE International', u'employer_QED Group',
       u'employer_Intergovernmental Relations Technical Committee (IGRTC)',
       u'employer_Management Systems International', u'employer_PATH',
       u'employer_Abt Associates', u'employer_One Acre Fund',
       u'employer_Kisii University', u'employer_Skills Geographic Kenya (SGK)',
       u'employer_Jomo Kenyatta University of Agriculture and Technology(JKUAT)',
       u'employer_other']
    df2 = df[cols_to_keep]
    df2 = df2.drop_duplicates(['description'])
    return df2

def get_original_data():
    df = pd.read_csv('data/all_jobs.csv')
    df['campaign_start_date'] = pd.to_datetime(df['campaign_start_date'])
    df['campaign_end_date'] = pd.to_datetime(df['campaign_end_date'])
    return df

def generate_recommendation(n, train, df_orig, top_5):
    idx = list(train.index.values).index(n)
    print 'train index {}'.format(idx)
    # idx_select = train.index.values == n
    original_index = train.iloc[idx,:]['Unnamed: 0']
    print original_index
    print train.iloc[idx,:]['title']
    start_date = df_orig.iloc[original_index,:]['campaign_start_date']
    end_date = df_orig.iloc[original_index,:]['campaign_end_date']
    print df_orig.iloc[original_index,:][['title','job_level','country','campaign_start_date','campaign_end_date','description','employer_name']]
    # print train.iloc[n,:]['title']
    print '\nTop similar jobs .......'

    top_5_orig_index = train.iloc[top_5[idx],:]['Unnamed: 0'].values
    print top_5[idx]
    print top_5_orig_index
    temp = df_orig.iloc[top_5_orig_index,:]
    print temp[temp['campaign_end_date']>start_date][['title','job_level','country','campaign_start_date','campaign_end_date','description','employer_name']]
    # print train.iloc[top_5[n],:]['title']
    print '----------------------------------------------\n'
    pass

if __name__=='__main__':
    df = get_feature_data()
    df_orig = get_original_data()
    punctuation = set(string.punctuation)
    stemmer = SnowballStemmer('english')
    df['desc_no_punc'] = df.description.map(lambda x: ''.join(ch for ch in x if x not in punctuation))
    df['desc_stemmed'] = df.description.map(lambda x: ' '.join([stemmer.stem(y) for y in x.decode('utf-8').split(' ')]))

    train = df
    # train, test = train_test_split(df,test_size=0.01, random_state=42)

    tf_vec = TfidfVectorizer(max_df=0.6, min_df=0.001, stop_words='english',ngram_range=(1,1),max_features=50)
    tf_vec = tf_vec.fit(train['desc_stemmed'].values)
    train_vector = tf_vec.transform(train['desc_stemmed'].values).toarray()
    # test_vector = tf_vec.transform(test['desc_stemmed'].values).toarray()

    train2 = train.drop(['description','desc_no_punc','desc_stemmed','Unnamed: 0','title'],axis=1)
    # test2 = test.drop(['description','desc_no_punc','desc_stemmed','Unnamed: 0'],axis=1)

    train2 = np.concatenate((train2.values,train_vector),axis=1)
    # test2 = np.concatenate((test2.values, test_vector),axis=1)

    scaler = StandardScaler()
    scaler = scaler.fit(train2)
    train_scaled = scaler.transform(train2)
    # test_scaled = scaler.transform(test2)

    pca = PCA(n_components=20)
    pca = pca.fit(train_scaled)
    train_reduced = pca.transform(train_scaled)
    # test_reduced = pca.transform(test_scaled)

    distances = cosine_similarity(train_reduced).T
    # distances = linear_kernel(train_reduced).T
    top_5 = []
    for i in xrange(len(train_reduced)):
        top = distances[i,:].argsort()[-6:-1][::-1]
        top_5.append(top)


    generate_recommendation(100, train, df_orig, top_5)

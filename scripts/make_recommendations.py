import pandas as pd
import numpy as np

def get_recommendation_data():
    df = pd.read_csv('../data/all_with_recs.csv')
    return df

def get_original_data():
    df = pd.read_csv('../data/all_jobs.csv')
    df['campaign_start_date'] = pd.to_datetime(df['campaign_start_date'])
    df['campaign_end_date'] = pd.to_datetime(df['campaign_end_date'])
    return df

def generate_recommendations(idx, df, df_orig):
    '''Function to generate a recommendation from a given index
    Finds the top valid similar jobs and prints them.
    INPUTS
    ---------------------------
    idx: Integer
    train: Pandas DataFrame
    df_orig: Pandas DataFrame
    '''
    # find original index
    original_index = df.iloc[idx,:]['orig_idx']
    # print 'original_index: {}'.format(original_index)
    print 'ORIGINAL:'
    print df.iloc[idx,:]['title']
    print df_orig.iloc[original_index,:][['title','job_level','country','campaign_start_date','campaign_end_date','description','employer_name']]
    # find start date for that add as don't want to recommend ads which have expired before this one started
    start_date = df_orig.iloc[original_index,:]['campaign_start_date']

    print '\n*************************'
    print 'Top similar jobs .......'
    top_5_orig_index = np.asarray([int(x) for x in df.iloc[idx,:]['top_recs'][1:-1].split(',')])
    print top_5_orig_index
    temp = df_orig.iloc[top_5_orig_index,:]
    print temp[temp['campaign_end_date']>start_date][['title','job_level','country','campaign_start_date','campaign_end_date','description','employer_name']]
    print '----------------------------------------------\n'
    pass

if __name__ == '__main__':
    # import data
    df_with_recs = get_recommendation_data()
    df_orig = get_original_data()

    # generate a recommendation
    for i in np.random.choice(df_with_recs.index, 5):
        generate_recommendations(i, df_with_recs, df_orig)

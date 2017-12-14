import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import log_loss, recall_score, precision_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def get_data():
    df = pd.read_csv('data/all_with_features.csv')
    cols_to_keep = [u'number_of_positions', u'required_work_experience_months', u'campaign_start_date_year',
       u'campaign_start_date_month', u'campaign_start_date_day',
       u'campaign_start_date_dow', u'desc_length',
       u'country_code_CI', u'country_code_DZ', u'country_code_KE',
       u'country_code_MU', u'country_code_MW', u'country_code_NG',
       u'country_code_PK', u'country_code_RW', u'country_code_SN',
       u'country_code_TZ', u'country_code_UG', u'country_code_ZA',
       u'country_code_nan', u'job_level_Entry-Level', u'job_level_Mid-Level',
       u'job_level_Senior', u'job_level_nan',u'location_Nairobi',
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
    X = df[cols_to_keep]
    y = df['campaign_duration']
    return X, y

def find_best_models(models, X, y):
    results = dict()
    for key, values in models.iteritems():
        print 'grid search for {}'.format(key)
        model = values[0]
        params = values[1]
        gridsearch = GridSearchCV(model, params, cv=5)
        gridsearch.fit(X, y)
        res = [gridsearch.best_estimator_, gridsearch.best_score_]
        results[key] = res
    return results

def save_submission(IDs, donate_probs):
    f = open('submission.csv', "w")
    f.write(",Made Donation in March 2007\n")
    for ID, prob in zip(IDs, donate_probs):
        f.write("{},{}\n".format(ID,prob[0]))
    f.close()

if __name__ == '__main__':
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model_list = {'rfc':[RandomForestRegressor(n_jobs=-1, oob_score=True),{'n_estimators': [50, 100,300],
                                                                            'max_depth': [5, 10 ,20],
                                                                            'max_features': [4,8,12]}],
                  'gbc':[GradientBoostingRegressor(),{'learning_rate': [0.01,0.001],
                                                        'n_estimators': [50,100,250],
                                                        'max_features': [4,8,12],
                                                        'max_depth': [3,5]}]}
                #   'mlp':[MLPClassifier(early_stopping=True),{'hidden_layer_sizes': [(500,),(500,2)],
                #                                              'activation': ['relu','logistic'],
                #                                              'learning_rate_init': [0.001, 0.0005],
                #                                              'max_iter':[200,400]}]}

    best_models = find_best_models(model_list, X_train, y_train)
    best_name = max(best_models.iterkeys(), key=(lambda key: best_models[key][1]))
    model = best_models[best_name][0]
    print 'Best score from cross_validation: {}'.format(best_models[best_name][1])
    model.fit(X_train, y_train)
    print 'Test score {}'.format(model.score(X_test, y_test))

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print('Feature Ranking:')
    for f in range(len(importances)):
        print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

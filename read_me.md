# Recommendation system for FUZU jobs website
FUZU is a career site connecting job seekers with opportunities in East Africa. FUZU has an Open API allowing developers to access all the open jobs as a JSON object.
I used this API to build a simple recommendation system using PCA and cosine similarity. The user also has the option to include the description and title which are vectorized using a TF-IDF vectorizer. This is an example of **item-item collaborative filtering**.


## Details
FUZU has a clear and simple Open API which allows the user to pull data on all open jobs postings. This is done using the *request* library

Several data fields have little or no data in so these are removed during the data cleaning stage. Also for categorical features such as country, location, employer, area of work, these are converted into dummy variables. Where there are many categories within a feature, categories with the majority entries are selected and dummified and all others added to 'other'. es

In the case of text fields such as 'title' and 'description', there is the option to vectorize these features and use them in the recommendation system. In this a tf-idf (term frequency - inverse document frequency) approach is used to select common but important words. The min and max document frequencies used are 0.005 and 0.5 respectively. The number of vectors outputted are defined by the user as they depend on the type of text field. For example, the description field is more complex then the title and therefore needs more vector components to correctly represent it.  

As there are several categorical features, the feature space is very large. Item-item collaborative filtering depends on finding the 'distance' or 'similarity' between job postings, therefore dimensionality needs to be reduced. For this I have used PCA and chosen 20 dimensions as this compromises between capturing variance in the data (~0.5 when running this code in May 2018) and mitigating the impact of increased dimensionality. I also used the cosine similarity metric to find similar posting as it is believed to be less sensitive to increased dimensionality. *scikit learn* library has been used for text vectorization and PCA.

The top 5 closest postings are found for each posting and saved. This allows the user to use the make_recommendations.py script to find recommendatiosn and evaluate the results without having to retrain the model.

## Files
The main source code files are found in the src folder. Code in development are found in the development folder.
**scripts folder**
- get_all_jobs.py: Basic script to call the API and download data.
- EDA.ipynb: Jupyter notebook to explore the data and initial features. Simple feature cleaning and engineering conducted in this notebook. Outputs data with features.
- model_recommendation.py; Script imports data, vectorizes text fields where required, applies PCA to the data and calculates cosine similarities. Also allows the user to find recommendations for a given posting to check recommendation performance. Export a csv with jobs and indexes of top recommendations
- make_recommendations.py: Script to import data and just make recommendations without having to retrain the model etc.

## To Do list/ Improvements
- Split main code block in model_recommendations into separate functions.
- Get user feedback about recommendation and potentially add weights to scaled features before PCA to make certain feature more important.

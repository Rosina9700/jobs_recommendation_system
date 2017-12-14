import requests
import pandas as pd

r = requests.get('http://www.fuzu.com/api/all_jobs')
r = r.json()
df = pd.DataFrame(r['fuzu_api'])
df.to_csv('data/all_jobs.csv',header=True,index=False,encoding='utf-8')

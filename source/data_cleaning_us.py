
import pandas as pd
import numpy  as np
import pickle
from source.utils import get_y_from_string, process_length, fill_database
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
np.random.seed(0)

df_raw = pd.read_excel('data/US Scales (Borello) - June 22.xlsx', 0)



RelevantFeatures=['INDUSTRY', 'LENGTH OF SERVICE (months)',
'IS THE WORKER IN A MANAGERIAL POSITION?',
'DOES THE EMPLOYER GUARANTEE WORK?', 'WHO DECIDES WHAT WORK IS DONE?',
'HOW IS THE WORK PERFORMED?', 'WHO SETS THE WEEKLY SCHEDULE',
'WHO SETS THE WORK HOURS',
'IS THE WORKER SUBJECT TO DISCIPLINARY ACTION OR SUPERVISION BY THE EMPLOYER?',
'HOW IS THE WORKER PAID?', 'DOES THE WORKER RECEIVE A BONUS?',
'RISK OF LOSS',
'IS THE WORK INCIDENTAL OR INTEGRAL TO THE EMPLOYER\'S BUSINESS?',
'WHAT IS THE DEGREE OF PERMANENCE?',
'IS THE WORKER ABLE TO WORK FOR OTHERS? (Ability to work --> NOT WHETHER WORK WAS DONE)',
'THE AMOUNT OF WORK DONE FOR THE HIRER',
'DOES THE WORK CONTRACT DESCRIBE THE WORKER\'S EMPLOYMENT STATUS',
'DOES THE WORKER HAVE AN INDEPENDENT BUSINESS AT THE TIME THE WORK IS BEING PERFORMED',
'IS THE BUSINESS INCORPORATED/LICENSED UNDER THE WORKER\'S NAME',
'IS THE INDEPENDENT BUSINESS OPERATION THE SAME AS THE WORK BEING PERFORMED FOR THE EMPLOYER?',
'DID THE WORKER OBTAIN ANY LICENSES OR ACCREDITATIONS UNDER HIS OR HER NAME?',
'DID THE WORKER LIST ADVERTISEMENTS UNDER HIS OR HER NAME',
'IS THE WORKER ABLE TO HIRE OTHERS?',
'DOES THE WORKER HAVE TO WEAR A UNIFORM? (Professional attire does not count i.e. business casual)',
'WHERE IS THE WORK PERFORMED?',
'WHO OWNS THE TOOLS AND EQUIPMENT USED FOR THE JOB?',
'WHETHER THE WORKER HOLDS THEMSELVES OUT AS BEING ENGAGED IN OCCUPATION OR BUSINESS DISTINCT FROM EMPLOYER?',
'DOES THE WORK REQUIRE A SPECIAL SKILL? (Only select yes/no if answer is explicitly stated in borello test)',
'WHAT IS PROCEDURE FOR TERMINATION?']

df = df_raw[RelevantFeatures]

df2 = df[~df['LENGTH OF SERVICE (months)'].isna()]

temp = []
for s in df2['LENGTH OF SERVICE (months)']:
    if isinstance(s, str):
        s=s.replace('1-3','2')
        s=s.replace('few hours','0')
        s=s.replace('+','')
        s=s.replace('>','')
        s=s.replace('<','')
        s=s.replace('~','')

        temp.append( [float(ss) for ss in s.split() if ss.isdigit()] )
    else:
        temp.append([s])
df2.loc[:,'LENGTH OF SERVICE (months)'] = np.array(temp)

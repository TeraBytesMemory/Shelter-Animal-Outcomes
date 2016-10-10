#!/usr/bin/env python
# coding: utf-8

from age_format import age_format, name_format, date_format, is_weekend, color_format

import pandas as pd

sex_map = {'Unknown': 0, 'Male': 1, 'Female': 2}
sex_outcome_map = {
    'Unknown': 0, 'Intact Male': 1, 'Intact Female': 2,
    'Neutered Male': 3, 'Spayed Female': 4
}
breed = {}
#color = {}

def data_format(data, train=True):
    global sex_map
    global sex_outcome_map
    global breed
    global color

    df = pd.read_csv(data)

    # test data には OutcomeTypeとOutcomeSubtypeは存在しない
    # colorを整形

    df['AnimalType'] = df.AnimalType.map({'Dog': 0, 'Cat': 1}).astype(int)

    df['Weekend'] = df.DateTime.map(is_weekend)
    df['DateTime'] = df.DateTime.map(date_format)

    if len(df.SexuponOutcome[ df.SexuponOutcome.isnull() ]) > 0:
        df.SexuponOutcome[ df.SexuponOutcome.isnull() ] = 'Unknown'

    df['Sex'] = df.SexuponOutcome.map(lambda x: x.split(' ')[-1])
    df['Sex'] = df.Sex.map(sex_map)

#    df['Intact'] = df.SexuponOutcome.map(lambda x: 0 if x == 'Unknown' else 1 if 'Intact' in x else -1)

    df['SexuponOutcome'] = df.SexuponOutcome.map(sex_outcome_map).astype(int)

    if len(df.AgeuponOutcome[ df.AgeuponOutcome.isnull() ]) > 0:
        df.AgeuponOutcome[ df.AgeuponOutcome.isnull() ] = '0'

    df['AgeuponOutcome'] = df.AgeuponOutcome.map(age_format).astype(int)

    #for i in (0, 1, 2): # median by Sex
    #    median_age = df['AgeuponOutcome'][ df.Sex == i ][ df.AgeuponOutcome != 0 ].median()
    #    df['AgeuponOutcome'][ df.Sex == i ][ df.AgeuponOutcome == 0 ] = median_age
    median_age = df['AgeuponOutcome'][ df.AgeuponOutcome != 0 ].median()
    df['AgeuponOutcome'][ df.AgeuponOutcome == 0 ] = median_age


    ''''
    df['Breed2'] = df.Breed.map(lambda x: x.split('/')[-1] if '/' in x else 'nan')
    df['Breed'] = df.Breed.map(lambda x: x.split('/')[0] if '/' in x else x)
    if train:
        breed_set = set(df['Breed'].values + df['Breed2'].values)
        breed = dict([(x, i) for i, x in enumerate(breed_set)])
        breed['nan'] = -1
    df['Breed'] = df.Breed.map(lambda x: breed[x] if x in breed.keys() else -1)
    df['Breed2'] = df.Breed2.map(lambda x: breed[x] if x in breed.keys() else -1)

    df['Color2'] = df.Color.map(lambda x: x.split('/')[-1] if '/' in x else 'nan')
    df['Color'] = df.Color.map(lambda x: x.split('/')[0] if '/' in x else x)
    if train:
        color_set = set(df['Color'].values)
        color = dict([(x, i) for i, x in enumerate(breed_set)])
        color['nan'] = -1
    df['Color'] = df.Color.map(lambda x: color[x] if x in color.keys() else -1)
    df['Color2'] = df.Color2.map(lambda x: color[x] if x in color.keys() else -1)
    #df['Color'] = df.Color.map(color_format)
    '''

    df['Breed'] = df.Breed.map(lambda x: len(x) if type(x) == str else 0)
    #df['Breed'] = df.Breed.map(lambda x: 1 if 'Mix' in x else 0)
    df['Color'] = df.Color.map(lambda x: len(x) if type(x) == str else 0)

    df['Name'] = df.Name.map(name_format)

#    df = df.drop(['Sex', 'AnimalType', 'Weekend'], axis=1)

    return df


def train_data_format(data):
    df = data_format(data)

    df['OutcomeType'] = df['OutcomeType'].map({
        'Adoption': 0 , 'Died': 1, 'Euthanasia': 2,
        'Return_to_owner': 3, 'Transfer': 4
    })
    y = df['OutcomeType'].values

    df = df.drop(['AnimalID', 'OutcomeType', 'OutcomeSubtype'], axis=1)
    x = df.values

    print(df.dtypes)

    return x, y


def test_data_format(data):
    df = data_format(data, train=False)

    ids = df['ID'].values
    df = df.drop(['ID'], axis=1)
    x = df.values

    return x, ids

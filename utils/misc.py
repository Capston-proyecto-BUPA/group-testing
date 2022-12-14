import os
import h2o
import pandas as pd


def save_model(leader_model, filename):
    print('Saving model...', end=' ', flush=True)
    model_path = h2o.save_model(model=leader_model, path=filename, force=True)
    print('done saving at: {}'.format(model_path))



def change_dates_to_ints(df):
    # date columns is now numerical
    # create 3 new columns called Year, Month and Day
    df.date = df.date.astype(str)
    df[['Year', 'Month', 'Day']] = df.date.str.split('-', expand=True)
    df.Month = df.Month.astype(float)
    df.Day = df.Day.astype(float)
    # This step still manual!
    df.Month = df.Month - 4
    df['date_num'] = df.Month * 30 + df.Day
    df.date = pd.to_datetime(df.date, format = "%Y-%m-%d")

    # Create relative dates
    df['relative_date'] = 0
    df['consecutive_date'] = 0
    for center in df['test_center'].unique():
        df.loc[df['test_center'] == center, 'relative_date'] = df.loc[df['test_center'] == center, 'date_num'] - df.loc[df['test_center'] == center, 'date_num'].min()
        df.loc[df['test_center'] == center, 'consecutive_date'] = list(range(len(df.loc[df['test_center'] == center, 'consecutive_date'])))

    return df



def generate_train_test(df,filterdate):
    # train = df.loc[df['Month'] == 0]
    # val = df.loc[df['Month'] == 1]
    train = df[df['date']<filterdate]
    val = df[df['date']>filterdate]

    # train.drop(columns=['Month', 'Day'], inplace=True)
    # val.drop(columns=['Month', 'Day'], inplace=True)

    return train, val


def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        print(f'{path} already exists')
import pdb
import numpy as np
import pandas as pd

DAYS_SINCE_POS = [1, 5, 10, 100, 500, 1000]

PATH_GEO = 'data/GeoInfo.csv'

def calculate_features(df, gap):
    """
    calculates the features for the data given a gap.
    Appends a column with the target for each datum.
    
    input: df contains the information for all
    the dates for all the test centers.

    returns: features a df where the rows are data points,
    the columns the features and the last column the targets.
    """
    # remove really historic feats (?)
    # prevalences of each test center
    #df["prevalence"] = df["positive"] / df["tests"]
    df["prevalence_day"] = df[f"lag_{gap}_pos_day"]/ df[f"lag_{gap}_tst_day"]
    df["prevalence_accum"] = df[f"lag_{gap}_pos"] / df[f"lag_{gap}_tst"]
    # global prevalence per date
    df["prevalence_total"] = df[f"lag_{gap}_totpos"] / df[f"lag_{gap}_tottst"]

    df["diff_1_pos"] = df[f"lag_{gap}_pos"] - df[f"lag_{gap + 1}_pos"]
    # df["diff_2_pos"] = df[f"lag_{gap + 1}_pos"] - df[f"lag_{gap + 2}_pos"]
    
    df["diff_1_tst"] = df[f"lag_{gap}_tst"] - df[f"lag_{gap + 1}_tst"]
    # df["diff_2_tst"] = df[f"lag_{gap + 1}_tst"] - df[f"lag_{gap + 2}_tst"]

    # df["diff_change_1_pos"] = df.diff_1_pos / df.diff_2_pos
    # df["diff_change_1_tst"] = df.diff_1_tst / df.diff_2_tst

    df["change_1_pos"] = df[f"lag_{gap}_pos"] / df[f"lag_{gap + 1}_pos"]
    # df["change_2_pos"] = df[f"lag_{gap + 1}_pos"] / df[f"lag_{gap + 2}_pos"]

    df["change_1_tst"] = df[f"lag_{gap}_tst"] / df[f"lag_{gap + 1}_tst"]
    # df["change_2_tst"] = df[f"lag_{gap + 1}_tst"] / df[f"lag_{gap + 2}_tst"]

    extra_features = []
    for pos in DAYS_SINCE_POS:
        df[f"days_since_{pos}_pos"] = (df[f"positive_{pos}_date"] - df.date).astype("timedelta64[D]")
        df.loc[df[f"days_since_{pos}_pos"] < gap, f"days_since_{pos}_pos"] = np.nan
        extra_features.append(f"days_since_{pos}_pos")

    # geographic fts
    info_geo = pd.read_csv(PATH_GEO)
    info_geo['Institución'] = info_geo['Institución'].replace(' ','',regex=True)
    info_geo.rename(columns={"Institución": 'test_center'}, inplace=True)
    info_geo = info_geo[~info_geo.duplicated()]
    info_geo = info_geo.dropna()
    df = pd.merge(df, info_geo, on=['test_center'])
    all_fts = [x for x in info_geo.columns[1:]]
    
    df["gap"] = gap

    # rename changing column names (relative to gap)
    df["lag_pos"] = df[f"lag_{gap}_pos"]
    df["lag_tst"] = df[f"lag_{gap}_tst"]

    df["lag_totpos"] = df[f"lag_{gap}_totpos"]
    df["lag_tottst"] = df[f"lag_{gap}_tottst"]

    # define target
    df["target"] = df["positives_accum"] / df["tests_accum"]

    features = [
        "date",
        "relative_date",
        "lag_pos",
        "lag_tst",
        "lag_totpos",
        "lag_tottst",
        "prevalence_accum",
        "prevalence_total",
        'diff_1_pos',
        'diff_1_tst',
        'change_1_pos',
        'change_1_tst']
    features_2 = [
        "total_tests",
        "tests_accum",
        "tests",
        "test_center",
        "gap",
        "target"
    ]
    
    features.extend(extra_features)
    features.extend(all_fts)
    features.extend(features_2)

    return df[features]


def gather_features(org_train_df, test_df, center_test_days, max_test_days):
    """
    Iterates over each posible gap and concatenates the features
    to produce the overall train_feats and test_feats
    """
    # fix consecutive dates, start from 1 now
    test_df.consecutive_date = test_df.consecutive_date + 1
    org_train_df.consecutive_date = org_train_df.consecutive_date + 1

    center_test_days_map = center_test_days.to_dict()
    center_org_train_days = org_train_df.groupby('test_center')['date'].count()
    center_org_train_days_map = center_org_train_days.to_dict()
    # extract the train ground-truth df (train_gt)
    comp_col = org_train_df.test_center.map(center_org_train_days) - org_train_df.test_center.map(center_test_days_map)
    train_df = org_train_df.loc[org_train_df.consecutive_date <= comp_col]
    val_df = org_train_df.loc[org_train_df.consecutive_date > comp_col]
    # TODO: change this for a more elegant version!
    # remove institutes that don't have enough training dates
    keep_centers = train_df.test_center.unique()
    val_df = val_df.loc[np.isin(val_df.test_center, keep_centers)]
    #test_df = test_df.loc[np.isin(test_df.test_center, keep_centers)]
    #org_train_df = org_train_df.loc[np.isin(org_train_df.test_center, keep_centers)]
    # rename the absolute dates of the val_df
    center_train_days = train_df.groupby('test_center')['date'].count().to_dict()
    val_df.consecutive_date = val_df.consecutive_date - val_df.test_center.map(center_train_days)
    test_df.consecutive_date = test_df.consecutive_date - test_df.test_center.map(center_org_train_days_map)
    # init empty feat dfs
    train_feats = pd.DataFrame()
    val_feats = pd.DataFrame()
    all_train_feats = pd.DataFrame()
    test_feats = pd.DataFrame()
    # iterate over each gap and append the features
    for gap in range(1, max_test_days+1):
        # in the dfs consider only the test_centers that have this gap available
        keep_centers = center_test_days.index[center_test_days.values >= gap]
        filtered_train = train_df.loc[np.isin(train_df.test_center.values, keep_centers)]
        filtered_val = val_df.loc[np.isin(val_df.test_center.values, keep_centers)]
        filtered_test = test_df.loc[np.isin(test_df.test_center.values, keep_centers)]
        keep_centers = center_org_train_days.index[center_org_train_days.values >= gap]
        filtered_org_train = org_train_df.loc[np.isin(org_train_df.test_center.values, keep_centers)]
        # extract the current val and test df according to the current gap

        current_val = filtered_val.loc[filtered_val.consecutive_date == gap]
        current_test = filtered_test.loc[filtered_test.consecutive_date == gap]
        # calculate the feats for each set
        these_train_feats = calculate_features(filtered_train, gap)
        these_val_feats = calculate_features(current_val, gap)
        all_these_train_feats = calculate_features(filtered_org_train, gap)
        these_test_feats = calculate_features(current_test, gap)
        # add the feats of the current gap to the complete df
        train_feats = pd.concat([train_feats, these_train_feats], ignore_index=True)
        val_feats = pd.concat([val_feats, these_val_feats], ignore_index=True)
        all_train_feats = pd.concat([all_train_feats, all_these_train_feats], ignore_index=True)
        test_feats = pd.concat([test_feats, these_test_feats], ignore_index=True)
    return train_feats, val_feats, all_train_feats, test_feats

def temporal_columns(train_df, test_df, center_test_days, max_test_days):
    test_df_aux = test_df[['date', 'test_center', 'tests', 'positive', 'Year',
                           'Month', 'Day', 'date_num', 'relative_date',
                           'consecutive_date']]
    test_df_noaux = test_df[['date', 'test_center', 'positives_accum', 'tests_accum', 'total_positive',
                             'total_tests']]
    df_panel = pd.concat((train_df, test_df_aux))  # create maxi df
    # iterate over each gap and append the features
    for gap in range(1, max_test_days+2):
        # in the dfs consider only the test_centers that have this gap available
        keep_centers = center_test_days.index[center_test_days.values >= gap]
        filtered_df = df_panel.loc[np.isin(df_panel.test_center.values, keep_centers)]
        df_panel[f'lag_{gap}_pos'] = filtered_df.groupby(
            'test_center')['positives_accum'].shift(gap)
        df_panel[f'lag_{gap}_tst'] = filtered_df.groupby(
            'test_center')['tests_accum'].shift(gap)
        df_panel[f'lag_{gap}_totpos'] = filtered_df.groupby(
            'test_center')['total_positive'].shift(gap)
        df_panel[f'lag_{gap}_tottst'] = filtered_df.groupby(
            'test_center')['total_tests'].shift(gap)
        df_panel[f'lag_{gap}_pos_day'] = filtered_df.groupby(
            'test_center')['positive'].shift(gap)
        df_panel[f'lag_{gap}_tst_day'] = filtered_df.groupby(
            'test_center')['tests'].shift(gap)
    
    for positive in DAYS_SINCE_POS:
        df_panel = df_panel.merge(
                        df_panel[df_panel.positives_accum >= positive].groupby(
                        'test_center')['date'].min().reset_index().rename(
                        columns={'date': f'positive_{positive}_date'}),
                        on='test_center', how="left")  
    
    # split df into sets
    final_train_df = df_panel.loc[np.isin(df_panel.date, train_df.date)]
    aux_test_df = df_panel.loc[np.isin(df_panel.date, test_df.date)]
    
    final_test_df = aux_test_df.merge(test_df_noaux, on=['date', 'test_center'], how='outer')
    final_test_df = final_test_df.assign(positives_accum_x=final_test_df['positives_accum_y'])
    final_test_df = final_test_df.assign(tests_accum_x=final_test_df['tests_accum_y'])
    final_test_df = final_test_df.assign(total_positive_x=final_test_df['total_positive_y'])
    final_test_df = final_test_df.assign(total_tests_x=final_test_df['total_tests_y'])

    final_test_df = final_test_df.rename(columns={'positives_accum_x': 'positives_accum'})
    final_test_df = final_test_df.rename(columns={'tests_accum_x': 'tests_accum'})
    final_test_df = final_test_df.rename(columns={'total_positive_x': 'total_positive'})
    final_test_df = final_test_df.rename(columns={'total_tests_x': 'total_tests'})

    final_test_df = final_test_df.drop(columns=['positives_accum_y', 'tests_accum_y', 'total_positive_y', 'total_tests_y'])
    return final_train_df, final_test_df

def remove_nans(df, value=0):
    df.fillna(value, inplace=True)
    return df

def get_features(train_df, test_df, sliding_window=False):
    """
    input: train_df and test_df which have the data for each
    test_center for all the dates belonging to the set.

    options:
    sliding_window: if true then we have to obtain a df for
    the training data for each window that fits into the 
    complete training data. We then proceed to concatenate the
    features for each sub train df below each other as the final
    train_feats

    returns: training (train_feats) and test (test_feats)
    feats ready for H2O, df where each row is a datum,
    the columns are the feats and the last column is the target.
    """
    # TODO: (to be done later add sliding window)
    # iterate over the train_dfs from each window
    
    # filter test_centers where window fits (?)

    # we assume that for training we want the same number of
    # test days as those available in the test_df
    center_test_days = test_df.groupby('test_center')['date'].count()
    max_test_days = center_test_days.max()
    # calculate lag and days since feats
    train_df, test_df = temporal_columns(train_df, test_df,
                                         center_test_days, max_test_days)
    # calculate the feats for each set
    train_feats, val_feats, all_train_feats, test_feats = gather_features(train_df, test_df,
                                                                          center_test_days,
                                                                          max_test_days)
    # postprocess the data (remove nans)
    train_feats = remove_nans(train_feats)
    val_feats = remove_nans(val_feats)
    all_train_feats = remove_nans(all_train_feats)
    test_feats = remove_nans(test_feats)
    return train_feats, val_feats, all_train_feats, test_feats
import os
import random
import argparse
import pandas as pd 

def create_df(config):
    print("entro?")
    geoinfo = config.geoinfo
    excelinfo = config.excelinfo
    filterdate = config.filterdate
    test_prev = config.test_prevalence
    #savepath = config.savepath
    info_geo = pd.read_csv(geoinfo)
    uniandes= pd.read_excel(excelinfo) 
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    info_geo['Institución']=info_geo['Institución'].replace(' ','',regex=True)
    info_geo=info_geo[~info_geo.duplicated()]
    info_geo=info_geo.dropna()
    merged_data=pd.merge(uniandes,info_geo,on=['Institución'])

    nuevo_orden=list(merged_data.columns)
    nuevo_orden.remove('Resultado')
    nuevo_orden.append('Resultado')
    merged_data = merged_data.reindex(columns=nuevo_orden)

    merged_data=merged_data[merged_data['Resultado']!='Inválido']
    merged_data.loc[:,'Resultado']=merged_data['Resultado'].replace('positivo','Positivo',regex=True)
    merged_data.loc[merged_data['Resultado']=='Negativo','Resultado']=0
    merged_data.loc[merged_data['Resultado']=='Positivo','Resultado']=1
    merged_data=merged_data[merged_data['Resultado']!='Indeterminado']
    merged_data.reset_index(inplace = True, drop = True) 


    merged_data['filter']=merged_data['Fecha recepción'].astype(str)+merged_data['Institución']
    filter_sum=merged_data.groupby(['filter'])['Resultado'].sum().values
    filter_keys=merged_data.groupby(['filter'])['Resultado'].sum().keys()
    filter_count=merged_data.groupby(['filter'])['Resultado'].count().values
    sum_dict = dict(zip(filter_keys,filter_sum))
    count_dict = dict(zip(filter_keys,filter_count))
    merged_data = merged_data[~merged_data.duplicated(subset='filter')]
    merged_data['sum']=merged_data['filter']

    merged_data['filter'].replace(count_dict,inplace=True)
    merged_data['sum'].replace(sum_dict,inplace=True)
    merged_data.reset_index(inplace = True, drop = True) 

    merged_data=merged_data[merged_data['Resultado']!='Indeterminado']

    filtro=merged_data[['Fecha recepción','Institución','filter','sum']].copy()

    if filterdate is not None:
        print('filtering data by date ' + filterdate)
        train=filtro[filtro['Fecha recepción']<filterdate].copy()
        val=filtro[filtro['Fecha recepción']>filterdate].copy()
        train=train[train.duplicated(subset='Institución',keep=False)]
        not_prev_data=list(set(val['Institución'].unique()) - set(train['Institución'].unique()))
        val=val[~val['Institución'].isin(not_prev_data)]
        
        prev=val['sum'].sum()/val['filter'].sum()
        random.seed(1234)
        if (prev > test_prev) & (test_prev!=-1): #Delete positives to decrease prevalence
            while prev > test_prev:
                delete_random=random.choice(val[val['sum']>0].index) 
                val.drop(delete_random,inplace=True)
                prev=val['sum'].sum()/val['filter'].sum()
        elif prev < test_prev: #Delete negatives to increase prevalence 
            while (prev < test_prev) & (test_prev!=-1):
                delete_random=random.choice(val[val['sum']==0].index) 
                val.drop(delete_random,inplace=True)
                prev=val['sum'].sum()/val['filter'].sum()

        print('The prevalence in test is: {}'.format(prev))
        filtro = pd.concat((train,val))
        filtro.reset_index(inplace = True, drop = True) 



    filtro.reset_index(inplace = True, drop = True) 
    keys=filtro.groupby('Fecha recepción')['sum'].sum().keys()
    p_total=dict(zip(keys,filtro.groupby('Fecha recepción')['sum'].sum().values))
    t_total=dict(zip(keys,filtro.groupby('Fecha recepción')['filter'].sum().values))
    filtro['total_tests']=filtro['Fecha recepción']
    filtro['total_positive']=filtro['Fecha recepción']
    filtro['total_tests'].replace(t_total,inplace=True)
    filtro['total_positive'].replace(p_total,inplace=True)
    filtro.rename(columns={'filter':'tests','sum':'positive','Fecha recepción':'date','Institución':'test_center'},inplace=True)
    # sort by dates at each institute
    filtro.sort_values(["test_center", "date"], inplace=True)
    # C: define cumulative data        
    filtro['positives_accum'] = filtro.groupby(["test_center"])["positive"].cumsum()
    filtro['tests_accum'] = filtro.groupby(["test_center"])["tests"].cumsum()
    return filtro , prev

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--geoinfo', type=str,
                    default='data/GeoInfo.csv')
    parser.add_argument('--excelinfo', type=str,
                    default='data/TestCenter.xlsx')
    parser.add_argument('--filterdate', type=str,
                    default='2020-05-08')
    
    config = parser.parse_args()
    config.test_prevalence = 0.05
    filtro, prev= create_df(config)
    print(filtro["tests_accum"])
    


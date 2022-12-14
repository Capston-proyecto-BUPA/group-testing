import h2o
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import precision_recall_curve
import pickle 

def calculate_efficiency_pool_thr(probs_one,groundtruth,poolsize,threshold):

    tests=0
    predicted_positives = 0
    real_positives = 0
    predicted_thr = (probs_one>threshold).astype(int)
    tests += predicted_thr.sum()
    detected_idx = np.where(predicted_thr==1)[0]
    predicted_positives += groundtruth[detected_idx].sum()
    real_positives += groundtruth[detected_idx].sum()
    complete_results = np.concatenate((probs_one.reshape(-1,1),groundtruth.reshape(-1,1)),axis=1)
    complete_results = np.delete(complete_results,detected_idx,axis=0)
    sorted_results = complete_results[np.argsort(complete_results[:, 0])[::-1]]
    groups = int(np.ceil(len(sorted_results)/poolsize))

    for g in range(groups):
        current = sorted_results[g*poolsize:poolsize+(poolsize*g)]
        tests +=1
        current_positives = current[:,1].sum()
        if current_positives > 0:
            tests += len(current)
            predicted_positives += current_positives

    efficiency = len(probs_one)/tests

    #print('Smart Pooling efficiency is '+ str(efficiency))

    return efficiency

def calculate_efficiency_best(probs_one,groundtruth):
    
    best_thr = []
    best_eff = []

    thresholds = np.arange(0,1,0.0005)
    pools = np.arange(2,13)
    for poolsize in tqdm(pools):
        efficiency_thr = []
        for thr in thresholds:
            tests=0
            predicted_positives = 0
            predicted_thr = (probs_one>thr).astype(int)
            tests += predicted_thr.sum()
            detected_idx = np.where(predicted_thr==1)[0]
            predicted_positives += groundtruth[detected_idx].sum()
            complete_results = np.concatenate((probs_one.reshape(-1,1),groundtruth.reshape(-1,1)),axis=1)
            complete_results = np.delete(complete_results,detected_idx,axis=0)
            sorted_results = complete_results[np.argsort(complete_results[:, 0])[::-1]]
            groups = int(np.ceil(len(sorted_results)/poolsize))

            for g in range(groups):
                current = sorted_results[g*poolsize:poolsize+(poolsize*g)]
                tests +=1
                current_positives = current[:,1].sum()
                if current_positives > 0:
                    tests += len(current)
                    predicted_positives += current_positives

            efficiency = len(probs_one)/tests
            efficiency_thr.append(efficiency)
        max_id_thr = np.argmax(efficiency_thr)
        best_thr.append(thresholds[max_id_thr])
        best_eff.append(efficiency_thr[max_id_thr])
    max_id_pool = np.argmax(best_eff)
    #print('Smart Pooling efficiency is '+ str(best_eff[max_id_pool]))

    return best_eff[max_id_pool],pools[max_id_pool],best_thr[max_id_pool]


def random_efficiency(gt,ventana):
    todos_tests = []
    todos_eficiencia = []
    ids = np.arange(len(gt))

    for i in range(2000):
        tests=0
        pacientes = 0
        random.shuffle(ids)
        gt=gt[ids]
        groups = int(np.ceil(len(gt)/ventana))

        for g in range(groups):
            actuales = gt[g*ventana:ventana+(ventana*g)]
            tests +=1
            positivos = actuales.sum()
            if positivos > 0:
                tests += len(actuales)
                pacientes += positivos

        eficiencia = len(gt)/tests
        todos_tests.append(tests)
        todos_eficiencia.append(eficiencia)

    #print('Dorfman Efficiency is '+ str(np.mean(todos_eficiencia)))
    #print('Std Efficiency is '+ str(np.std(todos_eficiencia)))
    return np.mean(todos_eficiencia)

def max_efficiency(gt,poolsize):
    original=gt
    indices_detectados = np.where(gt==1)[0]
    tests = gt[indices_detectados].sum()
    gt = np.delete(gt,indices_detectados,axis=0)
    groups = int(np.ceil(len(gt)/poolsize))
    tests += groups
    eficiencia = len(original)/tests
    print('Max Efficiency is '+ str(eficiencia))

    return eficiencia


def evaluate_best(model_leader,test,df, test_route='/data/TestCenter.xlsx'):
    preds =model_leader.predict(test)
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_test = test.as_data_frame().copy()
    df_test['pred_incidence'] = predictions.astype('float').squeeze()
    
    #Index(['Radicado', 'Fecha recepciÃ³n', 'InstituciÃ³n', 'Resultado'], dtype='object')
    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes.rename(columns={'Fecha recepciÃ³n': 'date', 'InstituciÃ³n': 'Institución', 'Resultado':'result'},inplace=True)
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Institución':'test_center'},inplace=True)
    uniandes=uniandes[uniandes['result']!= 'InvÃ¡lido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes=uniandes.loc[~uniandes.result.isna()]
    uniandes.reset_index(inplace = True, drop = True) 
    merged = pd.merge(df_test,uniandes,on=['date','test_center'])

    probability = merged['pred_incidence'].values
    groundtruth = merged['result'].values


    prevalence =100*(groundtruth.sum()/len(groundtruth))


    print('calculating efficiency on prevalence: {}'.format(prevalence))
    
    efficiency,poolsize,threshold = calculate_efficiency_best(probability,groundtruth)
    random_eff = random_efficiency(groundtruth,poolsize)
    maximum = max_efficiency(groundtruth,poolsize)
    print('Best in test - thr:{}, pool:{}, eff:{}'.format(threshold,poolsize,efficiency))


    return prevalence, efficiency, random_eff, maximum,threshold, poolsize, probability, groundtruth


def evaluate_thr_pool(model_leader,test,df,threshold, poolsize, test_route='data/TestCenter.xlsx'):
    preds =model_leader.predict(test)
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_test = test.as_data_frame().copy()
    df_test['pred_incidence'] = predictions.astype('float').squeeze()

    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes.rename(columns={'Fecha recepciÃ³n': 'date', 'InstituciÃ³n': 'Institución', 'Resultado':'result'},inplace=True)
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Institución':'test_center'},inplace=True)
    uniandes=uniandes[uniandes['result']!= 'InvÃ¡lido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes=uniandes.loc[~uniandes.result.isna()]
    uniandes.reset_index(inplace = True, drop = True) 
    merged = pd.merge(df_test,uniandes,on=['date','test_center'])

    probability = merged['pred_incidence'].values
    groundtruth = merged['result'].values
    prevalence =100*(groundtruth.sum()/len(groundtruth))



    print('calculating efficiency on prevalence: {}'.format(prevalence))
    
    efficiency = calculate_efficiency_pool_thr(probability,groundtruth, poolsize, threshold)
    random_eff = random_efficiency(groundtruth, poolsize)
    print('Test - thr: {}, pool: {}, eff:{}'.format(threshold,poolsize,efficiency))
    return prevalence, efficiency, random_eff

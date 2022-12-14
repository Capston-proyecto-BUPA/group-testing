import os
import h2o 
import json
import glob
import argparse
import itertools
import numpy as np
import pandas as pd
import os.path as osp
from h2o.automl import H2OAutoML 

# Custom Imports
from create_descriptors import create_df

from utils.misc import (save_model, change_dates_to_ints,
						generate_train_test, mkdirs)

from utils.get_features_gapallgeo import get_features
from utils.train_utils import train_models
from utils.eval_utils import evaluate_best, evaluate_thr_pool

def main(args, prev, train_model):

	args.test_prevalence = prev 
	filename = args.output_dir
	splits=2

	if prev <= 0.06:
		poolsize = 5
	elif prev <= 0.12:
		poolsize = 4
	else:
		poolsize = 3

	print('+++++++++++++++++++++++++PROCESSING DATA++++++++++++++++++++++++')

	df, prev = create_df(args)
	
	df = change_dates_to_ints(df)

	train, test = generate_train_test(df, args.filterdate)
	train, val, all_train, test = get_features(train, test)
	print("test", len(test))

	print('Uploading data to h2o...')
	train = h2o.H2OFrame(train, column_names=list(train.columns.astype(str)))
	val = h2o.H2OFrame(val, column_names=list(val.columns.astype(str)))
	all_train = h2o.H2OFrame(all_train, column_names=list(all_train.columns.astype(str)))
	test = h2o.H2OFrame(test, column_names=list(test.columns.astype(str)))
	mkdirs(args.output_dir)
	
	# Create output dirs
	experiment_name = args.output_dir.replace('/','_')+'_experiment'
	model_train_path = f'{args.output_dir}/train_model'
	model_all_train_path = f'{args.output_dir}/all_train_model'
	if train_model:
		models = train_models(train, val, experiment_name)

		print('+++++++++++++++++++++++++LEADERBOARD++++++++++++++++++++++++')
		best_model = models.leader
		save_model(best_model, model_train_path)
	else:
		best_model = h2o.load_model( f'{args.output_dir}/train_model/GBM_lr_annealing_selection_AutoML_1_20221214_122425_select_model')
		prevalence, efficiency, random_eff, maximum,threshold, pool, probability, groundtruth = evaluate_best(best_model,test,df, test_route='data/TestCenter.xlsx')

		prevalence, smart_pooling, dorfman = evaluate_thr_pool(best_model, test, df, threshold, pool)

		return prevalence, smart_pooling, dorfman, probability, groundtruth



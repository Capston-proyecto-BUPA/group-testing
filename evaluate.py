import os
import h2o 
import argparse
import numpy as np
from main import main
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, default='experiments/best_model',
                    help='Output directory')
parser.add_argument('--port', type=int, default=22222,
                    help='Port to connect h2o')
#parser.add_argument('--eval', action='store_true',
                   # help='Use this flag to evaluate a model')
parser.add_argument('--path-to-best', type=str, default=None,
                    help='Path to load best model, available when --eval flag is used.')
parser.add_argument('--path-to-val', type=str, default=None,
                    help='Path to load val model, available when --eval flag is used.')
parser.add_argument('--geoinfo', type=str,
                    default='data/GeoInfo.csv')
parser.add_argument('--excelinfo', type=str,
                    default='data/TestCenter.xlsx')
parser.add_argument('--filterdate', type=str,
                    default='2020-05-08')
parser.add_argument('--savegraph', action='store_true',
                    help='save individual graphs of experiments')
parser.add_argument('--poolsize', type=int, default=10, help='poolsize')


args = parser.parse_args()

h2o.init(port=args.port)

vec_prev, vec_smart, vec_dorfman  = [], [], []
prev_list = list(np.linspace(0.05, 0.25, 23)) # Evaluate on prevalences from 5% to 25%
# prev_list =[0.05] #Evaluate on original prevalence

main(args, 0.05, train_model = False)

diccionario = dict()



for prev in prev_list:
    diccionario[prev] = dict()
    diccionario[prev] = dict()
    prevalence, efficiency, random_eff, prob, groundtruth = main(args, prev, False)
    diccionario[prev]["prevalencia"] = prob.tolist()
    diccionario[prev]["resultado"] = groundtruth.tolist()
    vec_prev.append(prevalence)
    vec_smart.append(efficiency)
    vec_dorfman.append(random_eff)


h2o.cluster().shutdown()

with open('prevalencia.json', 'w') as file:
    json.dump(diccionario, file, indent=4)

print(vec_prev)
print(vec_smart)
print(vec_dorfman)


prevalence = vec_prev
vec_smartEff = vec_smart
vec_standardEff = vec_dorfman


output_dir = 'experiments/best_model'

vec_standardEff = 1/(np.array(vec_standardEff))
vec_smartEff = 1/(np.array(vec_smartEff))
One_efficiency_limit = np.ones(len(vec_smartEff))

# TestCenterDataset plot -----------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1,1,figsize=(7,6))

# Title
ax.margins(0)
ax.set_title('Metadata from test center',fontsize=12)


smart, = ax.plot(prevalence, vec_smartEff, label='Smart pooling', marker='o', color='black')

standard, = ax.plot(prevalence, vec_standardEff, label='Dorfman testing', linestyle='dotted', color='black')#, marker='o', markersize=4)
Eff_limitOne, = ax.plot(prevalence, One_efficiency_limit, label='Individual testing' , linestyle='dashed' , color='black', alpha=0.4)


# Solid fill
fill = ax.fill_between(prevalence, One_efficiency_limit, One_efficiency_limit+0.2, label='No improvement', facecolor='red', alpha=0.3)

# Axis labels
ax.set_xlabel('Prevalence [%]')
ax.set_ylabel('Expected number of tests per specimen')

box = ax.get_position()

fontP = FontProperties()
fontP.set_size('small')

# Legend inside the graph
ax.legend(handles=[fill, smart, standard, Eff_limitOne], loc='lower right',fancybox=True, shadow=True, ncol=1,prop=fontP, frameon=False)

ax.set(xlim=(5,25))
ax.set(ylim=(0,1.1))

plt.show()

# pdb.set_trace()

fig.savefig(os.path.join(output_dir,'TestCenter2.pdf'), dpi=600) # Route and name of the output file containing the generated plot

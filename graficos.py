import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

prevalence = [4.624277456647398, 5.835543766578249, 6.476683937823833, 7.614213197969544, 7.614213197969544, 7.614213197969544, 7.614213197969544, 7.614213197969544, 12.050359712230216, 13.057851239669422, 13.837209302325581, 14.901960784313726, 15.862708719851575, 16.24007060900265, 16.24007060900265, 16.24007060900265, 16.24007060900265, 20.327624720774388, 21.343596920923723, 22.188295165394404, 23.19422150882825, 24.093372238432682, 25.032481593763535]
vec_smartEff = [2.8833333333333333, 2.5821917808219177, 2.3975155279503104, 2.201117318435754, 2.201117318435754, 2.201117318435754, 2.201117318435754, 2.201117318435754, 1.7160493827160495, 1.596306068601583, 1.575091575091575, 1.48471615720524, 1.446979865771812, 1.4581724581724582, 1.4581724581724582, 1.4581724581724582, 1.4581724581724582, 1.3166666666666667, 1.3146274149034038, 1.2580025608194623, 1.2263779527559056, 1.2271099744245524, 1.1774604793472718]
vec_standardEff = [2.4211088818795052, 2.1664162758212906, 2.0557030763159623, 1.9136757712534758, 1.9136757712534758, 1.9136757712534758, 1.9136757712534758, 1.9136757712534758, 1.5271066495097578, 1.477548303943384, 1.441773295417175, 1.3942486234757598, 1.3548682731076234, 1.3412981794837509, 1.3412981794837509, 1.3412981794837509, 1.3412981794837509, 1.2079697319162201, 1.179941468629771, 1.159830946317147, 1.1354243342104287, 1.1162401938200788, 1.09604504017162]

dorfman_cesar = np.array([0.1983, 0.273, 0.3402, 0.386 , 0.4272, 0.4585, 0.496 , 0.5376,
       0.5764, 0.5912, 0.6188, 0.6368, 0.676, 0.7246, 0.8326, 0.9091,
       0.9868, 0.9967, 1.])
prevalencias_cesar = np.array([1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       15., 20., 25., 30., 31., 32.])

output_dir = 'experiments/best_model'

vec_standardEff = 1/(np.array(vec_standardEff))
vec_smartEff = 1/(np.array(vec_smartEff))
One_efficiency_limit = np.ones(len(vec_smartEff))

print(np.mean(np.abs(vec_smartEff - vec_standardEff)))

# TestCenterDataset plot -----------------------------------------------------------------------------------------------

fig, ax = plt.subplots(1,1,figsize=(7,6))

# Title
ax.margins(0)
ax.set_title('Metadata from test center',fontsize=12)

dorfman, = ax.plot(prevalencias_cesar, dorfman_cesar, label='Dorfman', linestyle='dotted', color='black',  marker='o')


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
ax.legend(handles=[fill, smart, standard, Eff_limitOne, dorfman], loc='lower right',fancybox=True, shadow=True, ncol=1,prop=fontP, frameon=False)

ax.set(xlim=(5,25))
ax.set(ylim=(0,1.1))

plt.show()

# pdb.set_trace()

fig.savefig(os.path.join(output_dir,'TestCenter.pdf'), dpi=600) # Route and name of the output file containing the generated plot
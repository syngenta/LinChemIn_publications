from linchemin.interfaces.facade import facade
from route_distances.ted.distances import distance_matrix

from rdkit.Chem import rdChemReactions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json


def compute_nx_ged_matrix():
    """ To compute the distance matrix for 100 routes with the GED algorithm implemented in NetworkX using LinChemIn """
    print('converting routes...')
    # The input file is available here: https://zenodo.org/record/6275421#.YzKYsXZBxPb
    routes = json.loads(open('n5-routes.json').read())
    syngraph_routes, _ = facade('translate', 'az_retro', routes[:100], 'syngraph', 'monopartite_reactions',
                                parallelization=True, n_cpu=8)

    print('computing distance matrix...')
    nx_distance_matrix, _ = facade('distance_matrix', syngraph_routes, ged_method='nx_ged',
                                   ged_params={'reaction_fp': 'difference_fp',
                                               'reaction_fp_params': {
                                                   'fpType': rdChemReactions.FingerprintType.MorganFP},
                                               'reaction_similarity_name': 'tanimoto'},
                                   parallelization=True, n_cpu=8)

    print('calculation of distance matrix completed')
    nx_distance_matrix.to_csv("paper_nx_distance_matrix.csv", index=False, header=False)


def compute_apted_matrix():
    """ To compute the distance matrix for 100 routes with the APTED algorithm as developed by
        Genheden at al. J. Chem. Inf. Mod. 61(8), 3899–3907 (2021)
    """
    file = json.loads(open('n5-routes.json').read())
    apted_array = distance_matrix(file[:100], content="reactions")
    apted_dist_matrix = pd.DataFrame(apted_array)
    apted_dist_matrix.to_csv('paper_apted_distance_matrix.csv', index=False)


def plot_from_files():
    """ To build the correlation plot """
    nx_array = pd.read_csv('paper_nx_distance_matrix.csv', header=None).values
    nx_array_clean = np.delete(nx_array, 0, 0)
    nx_values = nx_array_clean[np.triu_indices(nx_array_clean.shape[0], k=1)]

    apted_array = pd.read_csv('paper_apted_distance_matrix.csv', header=None).values
    apted_values = apted_array[np.triu_indices(apted_array.shape[0], k=1)]

    values = pd.DataFrame(columns=['networkx', 'apted'])
    values['networkx'] = nx_values
    values['apted'] = apted_values
    fig = sns.jointplot(x='networkx', y='apted', data=values, alpha=0.2)
    fig.ax_joint.set_xlabel('Ged NetworkX')
    fig.ax_joint.set_ylabel('Apted')

    plt.show()

    corr = values.corr(method='spearman')
    print(corr)


if __name__ == '__main__':
    compute_nx_ged_matrix()
    compute_apted_matrix()
    plot_from_files()

import json
from route_distances.ted.distances import distance_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from linchemin.interfaces.facade import facade
import time
from rdkit.Chem import rdChemReactions


def estimate_coef(x, y):
    """ To estimate di coefficients of a linear regression """
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def nx_parallel_comp_efficiency_data(syngraph_routes):
    """ To collect efficiency data for NetworkX GED calculations with parallel computing """
    efficiency = pd.DataFrame(columns=['n_routes', 'method', 'time_minutes'])
    n_routes = [2, 5, 10, 20, 30]
    efficiency['n_routes'] = n_routes
    efficiency['method'] = ['nx_ged_parallel'] * len(n_routes)
    delta_time = []
    for n in n_routes:
        print(f'computing distance matrix for {n} routes')
        t_start = time.time()
        facade('distance_matrix', syngraph_routes[:n], ged_method='nx_ged',
               ged_params={'reaction_fp': 'difference_fp',
                           'reaction_fp_params': {
                               'fpType': rdChemReactions.FingerprintType.MorganFP},
                           'reaction_similarity_name': 'tanimoto'},
               parallelization=True, n_cpu=14)
        end_time = time.time()
        delta_time.append((end_time - t_start) / 60.0)
    efficiency['time_minutes'] = delta_time
    efficiency.to_csv('nx_ged_parallel_comp_efficiency.csv', index=False)


def nx_efficiency_data(syngraph_routes):
    """ To collect efficiency data for NetworkX GED calculations """
    efficiency = pd.DataFrame(columns=['n_routes', 'method', 'time_minutes'])
    n_routes = [2, 5, 10, 20, 30]
    efficiency['n_routes'] = n_routes
    efficiency['method'] = ['nx_ged'] * len(n_routes)
    delta_time = []
    for n in n_routes:
        print(f'computing distance matrix for {n} routes')
        t_start = time.time()
        facade('distance_matrix', syngraph_routes[:n], ged_method='nx_ged',
               ged_params={'reaction_fp': 'difference_fp',
                           'reaction_fp_params': {
                               'fpType': rdChemReactions.FingerprintType.MorganFP},
                           'reaction_similarity_name': 'tanimoto'})
        end_time = time.time()
        delta_time.append((end_time - t_start) / 60.0)
    efficiency['time_minutes'] = delta_time
    efficiency.to_csv('nx_ged_efficiency.csv', index=False)


def apted_efficiency_data():
    """ To collect efficiency data for APTED calculations """
    file = json.loads(open('n5-routes.json').read())
    efficiency = pd.DataFrame(columns=['n_routes', 'method', 'time_min'])
    n_routes = [2, 5, 10, 20, 30]
    efficiency['n_routes'] = n_routes
    efficiency['method'] = ['apted'] * len(n_routes)
    delta_time = []
    for n in n_routes:
        t_start = time.time()
        dist_mat = distance_matrix(file[:n], content="reactions")
        end_time = time.time()
        delta_time.append((end_time - t_start) / 60.0)
    efficiency['time_min'] = delta_time
    efficiency.to_csv('apted_efficiency.csv', index=False)


if __name__ == '__main__':
    # The input file is available here: https://zenodo.org/record/6275421#.YzKYsXZBxPb
    routes = json.loads(open('n5-routes.json').read())
    syngraph_routes, _ = facade('translate', 'az_retro', routes[:30], 'syngraph', 'monopartite_reactions',
                                parallelization=True, n_cpu=8)
    # Generating the efficiency data and writing them to csv files
    nx_efficiency_data(syngraph_routes)
    nx_parallel_comp_efficiency_data(syngraph_routes)
    apted_efficiency_data()

    # Concatenating the data
    data = pd.read_csv('apted_efficiency.csv')
    apted_time = np.array(data['time_minutes'])
    x = np.array(data['n_routes'])

    df_efficiency_nx_parallel = pd.read_csv('nx_ged_parallel_comp_efficiency.csv')
    data = pd.concat([data, df_efficiency_nx_parallel], ignore_index=True)

    df_efficiency_nx = pd.read_csv('nx_ged_efficiency.csv')
    data = pd.concat([data, df_efficiency_nx], ignore_index=True)
    print(data)

    ged_time = np.array(data.loc[data['method'] == 'nx_ged', 'time_minutes'])
    ged_parallel_time = np.array(data.loc[data['method'] == 'nx_ged_parallel', 'time_minutes'])
    # Linear regression estimation
    slope_apted = estimate_coef(x, apted_time)
    print("Estimated coefficients for APTED:\nb_0 = {}  \
          \nb_1 = {}".format(slope_apted[0], slope_apted[1]))

    slope_nx = estimate_coef(x, ged_time)
    print("Estimated coefficients for GED NetworkX:\nb_0 = {}  \
          \nb_1 = {}".format(slope_nx[0], slope_nx[1]))

    slope_nx_parallel = estimate_coef(x, ged_parallel_time)
    print("Estimated coefficients for GED NetworkX:\nb_0 = {}  \
          \nb_1 = {}".format(slope_nx_parallel[0], slope_nx_parallel[1]))
    # Plot
    plt.scatter(x, ged_time, marker='o', color='red', s=30, label="Ged NetworkX")
    nx_reg = slope_nx[0] + slope_nx[1] * x
    plt.plot(x, nx_reg, color='darkorange')

    plt.plot(x, ged_parallel_time, 'go', label='Ged NetworkX - Parallelization')
    nx_reg_p = slope_nx_parallel[0] + slope_nx_parallel[1] * x
    plt.plot(x, nx_reg_p, color='limegreen')

    apted_reg = slope_apted[0] + slope_apted[1] * x
    plt.plot(x, apted_reg, color='cyan')
    plt.plot(x, apted_time, 'bo', label='Apted')

    plt.legend(loc="upper left")
    plt.xlabel('Nr Routes')
    plt.ylabel('Time (min)')
    font_nx = {'color': 'red'}
    font_apted = {'color': 'blue'}
    font_nx_p = {'color': 'limegreen'}

    plt.text(2, 2, f'y = {round(slope_nx[0], 2)} + {round(slope_nx[1], 2)} x', fontdict=font_nx)
    plt.text(12, 2, f'y = {round(slope_nx_parallel[0], 2)} + {round(slope_nx_parallel[1], 2)} x', fontdict=font_nx_p)
    plt.text(20, 0.65, f'y = {round(slope_apted[0], 4)} + {round(slope_apted[1], 4)} x', fontdict=font_apted)
    plt.show()

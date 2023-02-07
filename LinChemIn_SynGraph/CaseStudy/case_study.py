from linchemin.interfaces.workflows import process_routes
from linchemin.rem.route_descriptors import is_subset
import matplotlib.pyplot as plt
from linchemin.cgu.translate import translator
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#e377c2',  # raspberry yogurt pink
    '#17becf',  # blue-teal
    '#8c564b',  # chestnut brown
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#FF0000',  # red
    '#0000FF',  # blue
    '#008000',  # green
    '#808080',  # grey

]


def generate_clustering_figure(clustering, dist_matrix, mp_syngraphs):
    """ To generate the 3D plot of the clustering """
    dist_array = dist_matrix.to_numpy(dtype=float)
    labels = clustering.labels_
    unique_labels = set(labels)
    print('Nr clusters =', len(unique_labels))
    cols = ['route_id', 'x', 'y', 'cluster']

    fig2 = go.Figure()
    smallest_dist = 1000
    for k in unique_labels:
        d_temp = pd.DataFrame(columns=cols)
        # Collecting the indices of the routes in cluster k
        cluster_routes = np.where(clustering.labels_ == k)[0].tolist()
        ids = [mp_syngraphs[i].source for i in cluster_routes]
        print(f'Routes in cluster {k}:{ids}')
        xy = dist_array[np.where(clustering.labels_ == k)]
        fig2.add_trace(go.Scatter3d(x=xy[:, 0], y=xy[:, 32], z=xy[:, 61], mode="markers",
                                    marker=dict(
                                        size=15,
                                        color=colors[k],
                                        line=dict(width=2,
                                                  color='DarkSlateGrey')
                                    ),
                                    name=str(k)))
        fig2.update_layout(
            legend_title_text='Cluster',
            font=dict(size=15))
        # Retrieving the ids of the routes n the cluster
        ids = [mp_syngraphs[i].source for i in cluster_routes]
        d_temp['route_id'] = ids
        d_temp['cluster'] = k
        # Slicing the n rows corresponding to the routes in the cluster
        xy = dist_array[np.where(clustering.labels_ == k)]
        d_temp['x'] = xy[:, 0]
        d_temp['y'] = xy[:, 1]
        x_center = d_temp['x'].sum() / len(d_temp['x'])
        y_center = d_temp['y'].sum() / len(d_temp['y'])

        # Identifying the "most representative" route for each cluster
        id_center = None
        for n, id in enumerate(ids):
            dist_x = abs(d_temp.loc[n, 'x'] - x_center)
            dist_y = abs(d_temp.loc[n, 'y'] - y_center)
            dist = dist_x + dist_y
            if dist < smallest_dist:
                id_center = id
        # Generating the pictures of the representative route for each cluster
        route_center = [g for g in mp_syngraphs if g.source == id_center]
        print('Central route = ', route_center[0].source)
        translator('syngraph', route_center[0], 'pydot_visualization', 'monopartite_reactions')

    fig2.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig2.update_layout(
        font=dict(size=15))
    fig2.show()


def main():
    mol = 'MOL004'
    # The routes predicted by three CASP tools are processed and clustered
    out = process_routes({f'../data/AZ_AIZINTHFINDER/{mol}_routes.json': 'az',
                          f'../data/IBM_RXN/{mol}.json': 'ibmrxn',
                          f'../data/MIT_ASKCOS/{mol}_routes.json': 'askcos'
                          },
                         functionalities=['clustering_and_d_matrix'],
                         out_data_model='monopartite_reactions',
                         parallelization=True, n_cpu=8)

    cluster_output = out.clustering
    # Plotting hdbscan dendrogram
    cluster_output.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    plt.show()
    # Checking if there are identical routes
    routes = out.routes_list
    n_routes = len([r.uid for r in routes])
    n_unique_routes = len({r.uid for r in routes})
    print('total nr routes = ', n_routes)
    print('nr of unique routes =', n_unique_routes)
    if n_unique_routes != n_routes:
        duplicates = []
        for r1 in routes:
            duplicates.extend([r1.source, r2.source] for r2 in routes if r2.uid == r1.uid and r2.source != r1.source)
        print(duplicates)

    # Searching for subsets
    for g1 in routes:
        for g2 in routes:
            if is_subset(g1, g2):
                print(f'{g1.source} is subset of {g2.source}')

    dist_matrix = out.distance_matrix
    dist_matrix_round = dist_matrix.astype('float64').round(2)
    # Generating the distance matrix heatmap
    fig = px.imshow(dist_matrix_round, text_auto=True, aspect="auto",
                    labels=dict(color="GED"),
                    x=[r.source for r in routes],
                    y=[r.source for r in routes]
                    )

    fig.show()
    # Generating the cluster visualization
    generate_clustering_figure(cluster_output, dist_matrix, routes)


if __name__ == '__main__':
    main()
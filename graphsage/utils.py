from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph

import scipy
from concurrent.futures import ThreadPoolExecutor

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

mean = lambda l: sum(l)/len(l)

def create_cur(G, method, class_map, num_classes):
    N = [n for n in G.nodes() if not (G.node[n]['test'] or G.node[n]['val'])]
    M = []
    H = []
    from time import time
    from tqdm import tqdm
    def calc_m(n1):
        return 0
        m = [[]] * num_classes
        for n2,real_n2 in enumerate(N_shuff[:1000]):
            s = 0
            for i in range(len(e[0])):
                if e[0][i] == 0:
                    continue
                s += 1/e[0][i] * (e[1][n1,i] - e[1][n2,i])**2
            lab = class_map[real_n2]
            if isinstance(lab, list):
                lab = [idx for idx,val in enumerate(lab) if val == 1]
            else:
                lab = [lab]
            for l in lab:
                m[l].append(s)
        m = sorted([mean(l) for l in m])
        return 1/(m[1] - m[0])

    def calc_h(n):
        N_ex = N_sub[:]
        N_ex.remove(n)

        h = S[n,n] - np.matmul(np.matmul(S[n][:,N_ex],
                                        S_inv[N_ex][:,N_ex]),
                               S[N_ex][:,n])
        return h

    if method == 'R':
        step = 10000
        for offset in range(0,len(N),step):
            N_sub = N[offset:offset+step]
            N_shuff = N_sub[:]
            np.random.shuffle(N_shuff)
            L = nx.linalg.laplacianmatrix.laplacian_matrix(G, N_sub)

            N_sub = list(range(len(N_sub)))

            # Calc M
            # e = scipy.sparse.linalg.eigsh(L.asfptype())
            with ThreadPoolExecutor(10) as threads:
                for res in tqdm(threads.map(calc_m,N_sub), total = len(N_sub)):
                    M.append(res)
            # for n1 in tqdm(N_sub):
            #     M.append(calc_m(n1))

            # Calc H
            L = L + np.eye(L.shape[0])/100 
            # S = scipy.sparse.linalg.inv(L)
            S = np.linalg.inv(L)
            S_inv = L

            with ThreadPoolExecutor(5) as threads:
                for res in tqdm(threads.map(calc_h,N_sub), total = len(N_sub)):
                    H.append(res)
            # for fn in tqdm(N_sub):
            #     H.append(calc_h(fn))

        curr = np.array([m + h for m,h in zip(M,H)]).squeeze()
    elif method == 'degree':
        curr = G.degree()
    elif method == 'd_centrality':
        curr = nx.algorithms.centrality.degree_centrality(G)
    elif method == 'c_centrality':
        curr = nx.algorithms.centrality.closeness_centrality(G)
    elif method == 'b_centrality':
        curr = nx.algorithms.centrality.betweenness_centrality(G, 10000)
    elif method == 'clustering':
        curr = nx.algorithms.cluster.clustering(G)
    elif method == 'none':
        return None
    return curr


def load_data(prefix, curr_method, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    # if os.path.exists(prefix + "-cur.npy"):
    #     cur_R = np.load(prefix + "-cur.npy")
    # else:
    #     cur_R = create_cur(G, class_map, num_classes, 'R')
    #     np.save(prefix + "-cur.npy", cur_R)
    curr = create_cur(G, curr_method, class_map, num_classes)

    return G, feats, id_map, walks, class_map, curr

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

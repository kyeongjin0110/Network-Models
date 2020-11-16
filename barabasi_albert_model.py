# Barabasi-Albert

import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import operator
from scipy.stats import pearsonr 
import collections
import warnings
import random
from smallworld import get_smallworld_graph

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"]= "0" # gpu 0

colors = [  
            '#666666',
            '#1b9e77',
            '#e7298a'
            ]

class InitModel():

    def __init__(self, n, k_over_2, beta, focal_node):
        self.n = n
        self.k_over_2 = k_over_2
        self.beta = beta
        self.focal_node = focal_node

        self.smallWorldGraph = get_smallworld_graph(self.n, self.k_over_2, self.beta)
        self.edges = getEdges(self.smallWorldGraph, self.k_over_2, focal_node=self.focal_node)

        # print(type(edges[0]))

        self.graph = nx.Graph()
        self.graph.add_edges_from(self.edges)

    def getGraph(self):
        return self.graph

    def getEdges(self):
        return self.edges

    def showGraph(self):
        nx.draw_networkx(self.graph, pos=nx.circular_layout(sorted(self.graph.nodes(), reverse=True)), node_size = self.n, with_labels=True)
        plt.show()
      # plt.savefig("BarabasiAlbertGraph.png", format="PNG")

    def showGraph2(self):
        draw_network(self.smallWorldGraph, self.k_over_2, focal_node=0)
        plt.show()

class BarabasiAlbertModel():

    def __init__(self, n, m, init_edges):
        self.n = n
        self.m = m

        newGraph = nx.barabasi_albert_graph(self.n, self.m) 
        barabasei_edges = []
        for line in nx.generate_edgelist(newGraph, data=False):
            line = line.split()
            edge = (int(line[0]), int(line[1]))
            barabasei_edges.append(edge)

        self.total_barabasei_edges = init_edges + barabasei_edges
        # print(self.total_barabasei_edges)
        self.graph = nx.Graph()
        self.graph.add_edges_from(self.total_barabasei_edges)

    def getGraph(self):
        return self.graph

    def getEdges(self):
        return self.total_barabasei_edges

    def getDegreeDistribution(self):
        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)  # degree sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color="b")

        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)

        plt.show()

    def getMeanDegreeDistribution(self):
        print(self.graph.degree())
        degreelist = list([d for n, d in self.graph.degree()])
        return float(sum(degreelist))/nx.number_of_nodes(self.graph)
   
    def getClusteringCoefficient(self):
        return nx.clustering(self.graph)

    def getMeanClusteringCoefficient(self):
         return nx.average_clustering(self.graph)
   
    def getDiameter(self):
        path_list = []
        for C in nx.connected_component_subgraphs(self.graph):
            path_list.append(nx.diameter(C))
        return path_list

    def showGraph(self):
        nx.draw_networkx(self.graph, pos=nx.circular_layout(sorted(self.graph.nodes(), reverse=True)), node_size = self.n, with_labels=True)
        plt.show()
      # plt.savefig("BarabasiAlbertGraph.png", format="PNG")
    
    def showPowerLawDistribution(self):
        k = dict(nx.degree(self.graph))

        fig, ax = plt.subplots()

        # generate histogram data
        y, x = np.histogram(list(k.values()), bins=max(k.values()) - min(k.values()))
        x = x[:-1]
        y = y.astype('float')

        # normalize axix y
        y /= np.sum(y)
        plt.plot(x, y, ls='', marker='.')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('k')
        plt.ylabel('P(k)')
        
        plt.show()


def bezier_curve(P0, P1, P2, n=20):
    t = np.linspace(0,1,20)
    B = np.zeros((n,2))
    for part in range(n):
        t_ = t[part]

        B[part,:] = (1-t_)**2 * P0 + 2*(1-t_)*t_*P1+t_**2*P2

    return B


def is_shortrange(i, j, N, k_over_2):
    distance = np.abs(i-j)

    return distance <= k_over_2 or N-distance <= k_over_2


def draw_network(G, k_over_2, R=10, focal_node=None, ax=None):

    """
    Draw a small world network.
    Parameters
    ==========
    G : network.Graph
        The network to be drawn
    R : float, default : 10.0
        Radius of the circle
    focal_node : int, default : None
        If this is given, highlight edges
        connected to this node.
    ax : matplotlib.Axes, default : None
        Axes to draw on. If `None`, will generate
        a new one.
    Returns
    =======
    ax : matplotlib.Axes
    """

    G_ = G.copy()

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(3,3))

    focal_alpha = 1

    if focal_node is None:
        non_focal_alpha = 1
        focal_lw = 1.0
        non_focal_lw = 1.0
    else:
        non_focal_alpha = 0.6
        focal_lw = 1.5
        non_focal_lw = 1.0

    N = G_.number_of_nodes()

    phis = 2*np.pi*np.arange(N)/N + np.pi/2

    x = R * np.cos(phis)
    y = R * np.sin(phis)

    points = np.zeros((N,2))
    points[:,0] = x
    points[:,1] = y
    origin = np.zeros((2,))

    col = colors

    ax.axis('equal')
    ax.axis('off')

    if focal_node is None:
        edges = list(G_.edges(data=False))
    else:
        focal_edges = [ e for e in G_.edges(data=False) if focal_node in e]
        G_.remove_edges_from(focal_edges)
        edges = list(G_.edges) + focal_edges

    for i, j in edges:

        phi0 = phis[i]
        phi1 = phis[j]
        dphi = phi1 - phi0

        if dphi > np.pi:
            dphi = 2*np.pi - dphi
            phi0, phi1 = phi1, phi0
            phi1 += 2*np.pi

        distance = np.abs(i-j)

        if i == focal_node or j == focal_node:
            if distance <= k_over_2 or N-distance <= k_over_2:
                this_color = col[2]
            else:
                this_color = col[1]
            this_alpha = focal_alpha
            this_lw = focal_lw
        else:
            this_color = col[0]
            this_alpha = non_focal_alpha
            this_lw = non_focal_lw

        if distance == 1 or N-distance == 1:

            these_phis = np.linspace(phi0, phi1,20)
            these_x = R * np.cos(these_phis)
            these_y = R * np.sin(these_phis)

        else:
            if is_shortrange(i,j,N,k_over_2):
                ophi = phi0 + dphi/2
                o = np.array([
                            0.6*R*np.cos(ophi),
                            0.6*R*np.sin(ophi),
                    ])
            else:
                o = origin
            B = bezier_curve(points[i],o,points[j],n=20)
            these_x = B[:,0]
            these_y = B[:,1]

        ax.plot(these_x, these_y,c=this_color,alpha=this_alpha,lw=this_lw)

    ax.plot(x,y,'o',c='k')

    return ax, edges


def getEdges(G, k_over_2, R=10, focal_node=None, ax=None):

    G_ = G.copy()

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(3,3))

    focal_alpha = 1

    if focal_node is None:
        non_focal_alpha = 1
        focal_lw = 1.0
        non_focal_lw = 1.0
    else:
        non_focal_alpha = 0.6
        focal_lw = 1.5
        non_focal_lw = 1.0

    N = G_.number_of_nodes()

    phis = 2*np.pi*np.arange(N)/N + np.pi/2

    x = R * np.cos(phis)
    y = R * np.sin(phis)

    points = np.zeros((N,2))
    points[:,0] = x
    points[:,1] = y
    origin = np.zeros((2,))

    col = colors

    ax.axis('equal')
    ax.axis('off')

    if focal_node is None:
        edges = list(G_.edges(data=False))
    else:
        focal_edges = [ e for e in G_.edges(data=False) if focal_node in e]
        G_.remove_edges_from(focal_edges)
        edges = list(G_.edges) + focal_edges

    return edges


if __name__ == '__main__':

    n = 1000 # number of nodes
    k_over_2 = 2 # degree of nodes
    beta = 0
    focal_node = 0

    m = 2 # Number of edges to attach from a new node to existing nodes

    # Declare InitModel Class
    initGraph = InitModel(n, k_over_2, beta, focal_node)

    # Visualization init graph
    # initGraph.showGraph()
    # initGraph.showGraph2()

    init_edges = initGraph.getEdges()

    # Declare BarabasiAlbertModel Class
    barabasi = BarabasiAlbertModel(n, m, init_edges)

    # i) Get degree distribution
    barabasi.getDegreeDistribution()
    meanDegreeDistribution = barabasi.getMeanDegreeDistribution()
    print("\nmeanDegreeDistribution ..")
    print(meanDegreeDistribution)

    # ii) Get clustering coefficient
    clusteringCoefficient = barabasi.getClusteringCoefficient()
    print("\nclusteringCoefficient ..")
    print(clusteringCoefficient)
    meanClusteringCoefficient = barabasi.getMeanClusteringCoefficient()
    print("\nmeanClusteringCoefficient ..")
    print(meanClusteringCoefficient)

    # iii) Get diameter
    diameter = barabasi.getDiameter()
    print("\ndiameter ..")
    print(diameter)

    # iv) Visualization graph
    # barabasi.showGraph()

    # v) Visualization power law distribution
    barabasi.showPowerLawDistribution()

    ############################################################################################################################


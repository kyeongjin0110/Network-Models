# Erdos-Renyi

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

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["CUDA_VISIBLE_DEVICES"]= "0" # gpu 0


class ErdosRenyiRandomGraphModel():

   def __init__(self, n, p):
      self.n = n
      self.p = p
      self.graph = nx.gnp_random_graph(n, p)

   def getGraph(self):
      return self.graph

   def getNumberOfConnectedComponent(self):
      return nx.number_connected_components(self.graph)

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
      Gcc = self.graph.subgraph(sorted(nx.connected_components(self.graph), key=len, reverse=True)[0])
      pos = nx.spring_layout(self.graph)
      plt.axis("off")
      nx.draw_networkx_nodes(self.graph, pos, node_size=15, node_color='green')
      nx.draw_networkx_edges(self.graph, pos, alpha=0.4, edge_color='gray')
      plt.show()
      # plt.savefig("ErdosRenyiRandomGraph.png", format="PNG")

   def showGraphs(self, graphs, p_list):
      fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
      ax = axes.flatten()

      for i in range(4):
         nx.draw_networkx(graphs[i], pos=nx.circular_layout(sorted(graphs[i].nodes(), reverse=True)), ax=ax[i], with_labels=False, node_size = self.n, node_color='green', edge_color='gray')
         ax[i].set_axis_off()
         ax[i].set_title('Random graph with probability {}'.format(p_list[i]))

      plt.show() 


if __name__ == '__main__':

   n = 1000 # number of nodes
   p = 0.1 # probability of edge creation

   # Declare ErdosRenyiRandomGraphModel Class
   erodsRenyi = ErdosRenyiRandomGraphModel(n, p)

   # Connected components with respect to the value of p
   # i) Get number of connected coponents
   componentN = erodsRenyi.getNumberOfConnectedComponent()
   print("\ncomponentN ..")
   print(componentN)

   # ii) Get degree distribution
   erodsRenyi.getDegreeDistribution()
   meanDegreeDistribution = erodsRenyi.getMeanDegreeDistribution()
   print("\nmeanDegreeDistribution ..")
   print(meanDegreeDistribution)
   print("\nc: mean degree")
   c = (n-1)*p
   print(c)

   # iii) Get clustering coefficient
   clusteringCoefficient = erodsRenyi.getClusteringCoefficient()
   print("\nclusteringCoefficient ..")
   print(clusteringCoefficient)
   meanClusteringCoefficient = erodsRenyi.getMeanClusteringCoefficient()
   print("\nmeanClusteringCoefficient ..")
   print(meanClusteringCoefficient)
   print("\nC: clustering coeff")
   C = p
   print(C)

   # iv) Get diameter
   diameter = erodsRenyi.getDiameter()
   print("\ndiameter ..")
   print(diameter) # ~ log n

   # v) Visualization graph
   # erodsRenyi.showGraph()

   # Visualization graphs by p
   p_list = [0.1, 0.4, 0.6, 0.8]
   graphs = []
   for i in p_list:
      erodsRenyi_by_p = ErdosRenyiRandomGraphModel(n, p)
      graphs.append(erodsRenyi_by_p.getGraph())
   erodsRenyi.showGraphs(graphs, p_list)

   ############################################################################################################################
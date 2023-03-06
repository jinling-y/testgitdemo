#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:13:53 2023

@author: guillermo.vera
"""

import warnings
import numpy as np
import networkx as nx
import hypernetx as hnx
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import hypernetx.algorithms.hypergraph_modularity as hmod
import random
from random import sample

from scipy.cluster import hierarchy
# from pyscopus import Scopus
from scipy.sparse import issparse, coo_matrix, dok_matrix, csr_matrix as sci

import plotly.figure_factory as ff

import pandas as pd         # 导入pandas模块
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random
import copy
import math
import pickle
import statistics

from collections import deque

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from sklearn.cluster import AgglomerativeClustering
from functools import reduce
import math
from random import sample
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster
from matplotlib import pyplot as plt
from networkx.algorithms import community




def Generte_Hypergraph(node_n,edge_m,edge_range):
    '''
   
    Parameters
    ----------
    node_n : int
        节点的个数.
    edge_m : int
        边的个数.
    edge_range : list
        每条边的限制包含节点的个数.

    Returns
    -------
    node_list : list
        超图的节点列表.
    edge_list : list
        超图的 边列表

    '''
    edge_list=[]
    node_list=list(range(node_n))
    # [a,b]表示随机确定每条边中节点的个数的范围
    
    
    for j in range(edge_m):
        k=random.sample(list(range(edge_range[0],edge_range[1])),1)
        edge_list.append(sample(node_list,k[0]))
        
    return node_list,edge_list



def calcu_similar(node_list,edge_list,degree_dic,degree_i_j_dic):
    '''
    

    Parameters
    ----------
    node_list : list
        超图的节点列表.
    edge_list : list
        超图的边列表.
    degree_dic : dict
        key为所有的节点，value为对应节点的度，也就是有多少个节点包含该节点.
    degree_i_j_dic : dict
        key为节点的所有2组合 (i,j)，并且i<j. value表示这个2组合的度，也就是多少个边同时包含这两个节点

    Returns
    -------
    similar_dic : dict
        key 为节点的所有2组合(i,j).并且i<j. value表示这两个节点的相似度，
        如果完全不相似，value结果为9999，如果完全想对称，完全一样，结果为0

    '''
    
    similar_dic={}
    for (i,j) in degree_i_j_dic.keys():
        if degree_i_j_dic[(i,j)]==0:
            similar_dic[(i,j)]=999
        else:
            similar_dic[(i,j)]=(degree_dic[i]+degree_dic[j]-2*degree_i_j_dic[(i,j)])/degree_i_j_dic[(i,j)]
            
    return similar_dic

def calcu_adjacent_dict(node_list,edge_list):
    '''
    

    Parameters
    ----------
    node_list : list
        超图的节点列表.
    edge_list : list
        超图的边列表.

    Returns
    -------
    degree_dic : dict
        key为所有的节点，value为对应节点的度，也就是有多少个节点包含该节点.
    degree_i_j_dic : dict
        key为节点的所有2组合 (i,j)，并且i>j.value表示这个2组合的度，也就是多少个边同时包含这两个节点

    '''
    

    
    degree_dic={}
    for i in node_list:
        d_i=0
        for each in edge_list:
            if i in each:
                d_i+=1
        degree_dic[i]=d_i
    degree_i_j_dic={}
    for i in node_list:
        for j in node_list:
            if i<j:
                d_i_j=0
                for each in edge_list:
                    if i in each and j in each:
                        d_i_j+=1
                degree_i_j_dic[(i,j)]=d_i_j
                
    return degree_dic,degree_i_j_dic
                



        
# LEXICAS=list(pd.read_csv('LEXICAS.csv').iloc[:,0])
# NL=len(LEXICAS)


# node_list = list(range(len(LEXICAS)))

# with open('Matrices.pkl', 'rb') as file:
#     Resultados = pickle.load(file)

# FL=Resultados['Frases']; B=Resultados['Matriz_interseccion']
# nFL=Resultados['Frases_repetidas']; tFL=Resultados['Frases_paper']
# C=Resultados['Matriz_proyeccion']; 

# edge_list = FL

# maxi = []
# for i in FL:
#     maxi.append(max(i))
# print(max(maxi))
# H = hnx.Hypergraph([{'a','b'},{'c','d'},{'a','b','c'}])

# node_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
# edge_list = [['a','b'],['c','d'],['a','b','c'],['a','b','d'],['e','f','g'],
#               ['a','b','d','e'],['c','f','d','a'],['f','g'],['e','g','d'],['g','e'],
#               ['e','f'],['a','h'],['i','h','j','k'],['j','e','g'],['i','j','k','b'],
#               ['d','k','e','j'],['k','h'],['l','m','n'],['n','e','f'],['l','h','g'],['m','c']]

# node_list = ['a','b','c','d','e']
# edge_list = [['a','b','d'],['b','d'],['a','d'],['a','c','d'],['b','c','d'],['e','a']]

# node_list = [0,1,2,3,4,5,6,7]
# edge_list = [[0,1,2],[4,3,2],[6,5,1],[1,6],[3,4,5,6],[2,3,6],[2,3,5],[1,4,6],[0,7],[0,1,7],[1,7]]

node_list,edge_list  = Generte_Hypergraph(100,80,[2,15])
degree_dic,degree_i_j_dic = calcu_adjacent_dict(node_list,edge_list)
similar_dic               = calcu_similar(node_list,edge_list,degree_dic,degree_i_j_dic)

derivada = []
derivada_2 = []
for i in node_list:
    for j in node_list:
        if i<j:
            derivada.append(similar_dic[(i,j)])
            if similar_dic[(i,j)]<998:
               derivada_2.append(similar_dic[(i,j)])

def matriz_sim(lista, threshold):
    n = int((1+np.sqrt(1+8*len(lista)))/2)
    derivative_matrix = np.zeros([n,n])
    community_matrix = np.zeros([n,n])
    k = 0 #contador
    for i in list(range(n)):
        #matrix.append([])
        for j in list(range(n-i)):
            if i==j+i:
                derivative_matrix[i][j+i] = 0
                community_matrix[i][j+i] = 1
            elif i<j+i+1:
                derivative_matrix[i][j+i] = lista[k+j-1]
                if lista[k+j-1] < threshold:
                    community_matrix[i][j+i] = 1
                else:
                    community_matrix[i][j+i] = 0
        k = k+j
    derivative_matrix = np.transpose(derivative_matrix) + derivative_matrix 
    community_matrix = np.transpose(community_matrix) + community_matrix - np.eye(n)
    return derivative_matrix, community_matrix



H_dict = {}
j = 0
for i in sorted(edge_list):
    j = j - 1
    H_dict[j] = sorted(i)
# print(H_dict)

# H_dict = {'0':['a','b'],'1':['c','d'],'2':['a','b','c']}

H = hnx.Hypergraph(H_dict)
HG = hmod.precompute_attributes(H)

# Communities_set_list = []
# for i in Communities:
#     Communities_set_list.append(set(i))
# Communities_set_list.remove({248})
# Communities_set_list.remove({609})
# Communities_set_list.remove({1597})
# q2 = hmod.modularity(HG,Communities_set_list)
# print(q2)
# K = hmod.kumar(HG)
# q = hmod.modularity(HG, K)
# print(q)
# L = hmod.last_step(HG, K)
# q1 = hmod.modularity(HG,L)
# print(q1)
# H.collapse_edges

# Añadir una hiperarista a un hipergrafo

# H.add_edge(edge)
# Añadir un nodo a una hiperarista
# H.hnx.add_node_to_edge('node','edge')

# Matriz de incidencia 
# Incidence_matrix = H.incidence_matrix()

# Matriz de adyacencia 
# Adjacency_matrix = H.adjacency_matrix()

# Devuelve matrices tipo csr, para cambiarlas a tipo npArray (matriz tipo numpy)
# Adjacency_matrix_np = Adjacency_matrix.toarray()


def derivative_matrix_of_a_hypergraph(hypergraph):
    Matrix_incidence_of_hypergraph = hypergraph.incidence_matrix()
    Matrix_incidence_of_hypergraph_np = Matrix_incidence_of_hypergraph.toarray()
    M = np.dot(Matrix_incidence_of_hypergraph_np,
                                               np.transpose(Matrix_incidence_of_hypergraph_np))
    n = len(M)
    derivative_matrix = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if M[i][j]==0:
                derivative_matrix[i][j] = -1
            else:
                derivative_matrix[i][j] = (M[i][i]+M[j][j]-2*M[i][j])/M[i][j]
    # G = nx.from_numpy_array(Matrix_adjacency_of_hypergraph_np)
    return derivative_matrix

def means_of_a_matrix(matrix):
    n = len(matrix)
    Har = []
    M = []
    k = 0
    for i in range(n):
        for j in range(n-i):
            if i<=j+i+1 and matrix[i][j]>0 and matrix[i][j]<998:
                Har.append(matrix[i][j])
                M.append(matrix[i][j])
            elif i<=j+i+1 and matrix[i][j]<998:
                M.append(matrix[i][j])
        k = k+j
    harmonic_mean = statistics.harmonic_mean(Har)
    normal_mean = statistics.mean(M)
    des_tipica = statistics.stdev(Har)
    return harmonic_mean, normal_mean, des_tipica


# hogwarts = hnx.HarryPotter()
# E = hnx.StaticEntity(data = hogwarts.data, labels = hogwarts.labels)
# ES = hnx.StaticEntitySet(E)
# H = hnx.Hypergraph(ES, static=True,name='Hogwats')
# hnx.draw(H)


# plt.figure(1)
# d = derivative_matrix_of_a_hypergraph(H)


derivative_matrix,community_matrix =matriz_sim(derivada, 0)

harmonic_media, normal_media, des_tipica = means_of_a_matrix(derivative_matrix)

paso_malla = 0.1
Number_communities_list = []
# Ratio_list = []
# Solitaria_list = []
cont = -1
# X = np.zeros([len(node_list),len(np.arange(0,harmonic_media+0.5*des_tipica,0.1))])
for threshold in np.arange(0,harmonic_media+0.5*des_tipica,paso_malla):
    # cont = cont + 1
    # cont2 = -1
    # m = d<= threshold
    # m = np.multiply(m,1)
    d,community_matrix =matriz_sim(derivada, threshold)
    community_graph = nx.from_numpy_array(community_matrix)
    communities = nx.connected_components(community_graph)
    n = nx.number_connected_components(community_graph)
    Number_communities_list.append(n)
    Communities = []
    for i in range(n):
        C = []
        community = list(next(communities))
        # print(community)
        for pos in community:
            C.append(node_list[pos])
        # print(C)
        Communities.append(C)
    
    # Palabras = []
    # for i in Communities:
    #     aux = []
    #     for j in i:
    #         aux.append(LEXICAS[j])
    #     Palabras.append(aux)
    # print(Palabras)
    
    # solitarias = [[]]
    # for i in Communities:
    #     if len(i)<=5:
    #         solitarias.append(i)
    # Ratio_list.append(n/len(solitarias))
    # Solitaria_list.append(len(solitarias))
    # Communities = []
    # k = 0
    # for i in range(n):
    #     C = []
    #     community = list(next(communities))
    #     # print(community)
    #     for pos in community:
    #         C.append(node_list[pos])
    #     # print(C)
    #     Communities.append(C)
    # # print(Communities)
    # for Community in Communities:
    #     for j in Community:
    #         cont2 = cont2 + 1
    #         X[cont2][cont] = k
    #     k = k+1    



d,community_matrix = matriz_sim(derivada,harmonic_media)
G = nx.from_numpy_array(community_matrix)
communities = nx.connected_components(G)
n = nx.number_connected_components(G)
print('Numero de comunidades', n)
plt.figure()
hnx.draw(H)


Communities = []
for i in range(n):
    C = []
    community = list(next(communities))
    # print(community)
    for pos in community:
        C.append(node_list[pos])
    # print(C)
    Communities.append(C)

# Palabras = [[101, 102],[221],[248],[349],[400],[530],[609],[669],[849],[980],[1033],[1235,1236,1237],[1270],[1395,1396],[1400,1399],[1409],[1413],[1442],[1461,1462],[1501],[1505],[1536],[1559],[1597],[1601,1602],[1632],[1666]]
# Palabras = []
# for i in Communities:
#     aux = []
#     for j in i:
#         aux.append(LEXICAS[j])
#     Palabras.append(aux)
# print(Palabras)

# solitude = []
# for i in Palabras:
#     if len(i)<=5:
#         solitude.append(i)
# print(solitude)        
# print('Las palabras solitarias son:', solitarias, 'y de las', n, 'comunidades que hay', len(solitarias), 'son solitarias')               
# HG = hmod.precompute_attributes(H)
# q = hmod.modularity(HG,C)
# print(q)
    
# print(n)
plt.figure()
plt.title("Número de comunidades vs Umbral")
plt.plot(np.arange(0,harmonic_media+0.5*des_tipica,paso_malla),Number_communities_list)
plt.xlabel("Umbral")
# plt.title("Número de comunidades solitarias vs Umbral")
# plt.plot(np.arange(0,harmonic_media+0.5*des_tipica,paso_malla),Solitaria_list)
# plt.xlabel("Umbral")

# # plt.xticks([harmonic_media],['Media armónica'],fontsize=8, weight='bold')
# plt.ylabel("Número de comunidades")
plt.grid()
plt.plot(harmonic_media*(np.ones(len(Number_communities_list))),Number_communities_list)


# plt.figure()
# diferencia = [e1 - e2 for e1, e2 in zip(Number_communities_list ,Solitaria_list)]
# plt.title("Diferencia entre total y solitarias vs Umbral")
# plt.plot(np.arange(0,harmonic_media+0.5*des_tipica,paso_malla),diferencia)
# plt.xlabel("Umbral")

# plt.figure()
# plt.title("Número de comunidades vs Número decomunidades solitarias")
# plt.plot(np.arange(0,harmonic_media+0.5*des_tipica,paso_malla),Ratio_list)
# plt.xlabel("Umbral")
# plt.ylabel("Ratio")
# names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l','m','n']
# fig = ff.create_dendrogram(X, orientation='left', labels=names)
# fig.update_layout(width=800, height=800)
# fig.show()


# X = np.random.rand(15, 12) # 15 samples, with 12 dimensions each
# fig = ff.create_dendrogram(X)
# fig.update_layout(width=800, height=500)
# fig.show()

# ytdist = np.array([662., 877., 255., 412., 996., 295., 468.,268., 400., 101.,
#                    754., 564., 138., 219., 869.])
Z = hierarchy.linkage(derivada, 'single')
plt.figure()
dn = hierarchy.dendrogram(Z, labels=node_list,orientation='top', count_sort='ascending')
plt.plot(list(range(len(Number_communities_list))),harmonic_media*(np.ones(len(Number_communities_list))))

# print(Communities)    

moda = statistics.mode(Number_communities_list)
while moda == len(node_list) or moda == 1:
    Number_communities_list.remove(moda)
    moda = statistics.mode(Number_communities_list)
    
    
print('media armónica:', harmonic_media, 'media normal:', normal_media,'desviación típica:', des_tipica,
      'moda de comunidades:', moda)
# scenes = {
#     0: ('FN','TH'),
#     1: ('TH','JV'),
#     4: ('JU','CH','BR','CN','FN','TH')}

# H1 = hnx.Hypergraph(scenes)
# hnx.draw(H1)
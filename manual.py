import os
import networkx as nx

#USES NVIDIA NUMBA CUDA JIT COMPILER FOR NUMERIC CALCULATIONS
from numba import jit

from pandas import read_csv
from scipy.spatial import distance
from shortpath import (array_to_graph, extract_path_info)
import itertools
from itertools import islice
import math
import _pickle as cPickle


df = read_csv('./systems.csv', usecols=[2,3,4,5])

import numpy as np
import sys

#route start point
start_point = [0,0,0]

#route destination point
end_point = [-47.125,-3.25,60.28125]

#deviation from direct path in LightYears (Initial Corridor Width)
max_deviation = 30

#max FrameShiftDrive Jump Range
fsdrange = 10

kpairs = 10
knn = 20
nbrs_threshold = 10.00
nbrs_threshold_step = 0.02
base_point = start_point

@jit
def createGraph(point_cloud, base_point, kpairs, knn, nbrs_threshold, nbrs_threshold_step):
    G = array_to_graph(point_cloud, base_point, kpairs, knn, nbrs_threshold, nbrs_threshold_step)
    return G

@jit
def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z
@jit
def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)
@jit
def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)
@jit
def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

@jit
def distance(p0,p1):
    return length(vector(p0,p1))
@jit
def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

@jit
def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)

#CALCULATE DISTANCE OF A POINT_A TO CLOSEST POINT_X
#WHERE POINT_X IS ON A LINE between START_POINT AND END_POINT
@jit
def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

@jit
def inFSDrange(candidates):
    if not candidates:
        return False
    for a, b in itertools.combinations(candidates, 2):
        if length(vector(a,b)) >= fsdrange:
            return False
    return True

@jit
def findCandidates(point_cloud, candidates, max_deviation):
    names=[]

    names.append( df.loc[(df['x']==candidates[0][0]) & (df['y']==candidates[0][1]), 'name'])
    names.append( df.loc[(df['x']==candidates[1][0]) & (df['y']==candidates[1][1]), 'name'])

    for point in point_cloud:
        distance, nearest_point = pnt2line(point, start_point, end_point)
        if distance < max_deviation:
           candidates = np.concatenate([candidates, np.array([point])])
           names.append(df.loc[(df['x'] == point[0]) & (df['y']==point[1]), 'name'])
    return candidates, names

@jit
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

if __name__ == "__main__":

    point_cloud = df.as_matrix(columns=['x','y','z'])
    candidates = np.array([end_point, start_point])
    candidates, names = findCandidates(point_cloud, candidates, max_deviation)

    if os.path.isfile("graph.pickle"):
        with open(r"graph.pickle", "rb") as graph_file:
            G = cPickle.load(graph_file)
    else:
        G = createGraph(candidates, 0, kpairs, knn, nbrs_threshold, nbrs_threshold_step)
        with open(r"graph.pickle", "wb") as graph_file:
            cPickle.dump(G, graph_file)

    paths= k_shortest_paths(G,1,0,10)

    valid_paths = []

    print("FOUND ROUTES","\n", paths)
    print("CHECKING JUMP CAPABILITY:")

    for path in paths:
        path_valid = 1
        i=0
        while i < len(path)-1:
            dist = distance(candidates[path[i]],candidates[path[i+1]])
            if dist>fsdrange:
                print("PATH:", path , " JUMP ", i, "EXCEEDS FSDRANGE. DISTANCE WAS ", dist)
                path_valid = 0
                break
            i = i + 1
        if path_valid==1:
            valid_paths.append(path)

    print("N SHORTEST ROUTES VALID FOR YOUR FSD RANGE ARE:")
    print(valid_paths)

    #distances, paths = extract_path_info(G, 0, return_path=True)

    #print("1" , "\t", names[1].values ,"\t", candidates[1], "\t", distances[1], "\t", paths[1])

# app/clustering/enums/enums.py

from enum import Enum


class HierarchicalLinkage(str, Enum):
    ward = "ward"
    complete = "complete"
    average = "average"
    single = "single"


class HierarchicalMetric(str, Enum):
    euclidean = "euclidean"
    l1 = "l1"
    l2 = "l2"
    manhattan = "manhattan"
    cosine = "cosine"
    precomputed = "precomputed"


class DBSCANMetric(str, Enum):
    euclidean = "euclidean"
    l1 = "l1"
    l2 = "l2"
    manhattan = "manhattan"
    cosine = "cosine"
    precomputed = "precomputed"


class DBSCANAlgorithm(str, Enum):
    auto = "auto"
    ball_tree = "ball_tree"
    kd_tree = "kd_tree"
    brute = "brute"

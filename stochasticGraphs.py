# %%
%%markdown
# Stochastic Processes in Graphs
##### Han Du
# %%
%%markdown

The purposes of this notebook is to study stochastic processes and algorithms on graphs.

I will represent my graphs as an adjacency matrix $G$.

$G_{i,j} = 1$ if there is an interaction between node i and node j.

$G_{i,i}$ represents the value stored by node i.

## Information Propagation in a Stochastic Network from Single Source

We start with studying the speed of information propagation in a graph where values of nodes are updated by the rule:

$G_{i,j} = 1 \implies \max{(G_{i,i},G_{j,j})} \rightarrow G_{i,i}, G_{j,j}$

In this section we study random graphs $G = (N, p)$ where $N$ is fixed number of vertices and $p$ represents the probability of an edge being included.


# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce

n = 10
p = 0.5
upperDiagIndices = np.triu_indices(n, 1)

# graph initialization




def randomizeEdges(G, n, p):
    m = map((lambda x: x > p), np.random.normal(size = int((n-1) * n / 2)))
    edges = np.array([1 if e else 0 for e in m])
    G[upperDiagIndices] = edges
    return G


def randomAccumulate(G, n, p):
    randIdx = np.random.randint(low = 0, high = n)
    G[randIdx, randIdx] += 1
    return G


def drawGraph(G):
    nxG = nx.from_numpy_matrix(G)
    labelDict = dict(zip(nxG, np.diag(G).astype(int).tolist()))
    nx.draw(nxG, labels = labelDict)
    ax = plt.gca()
    ax.set_aspect('equal')


def oneStepPropagation(G):
    changed = []
    for i, row in enumerate(G):
        for j, column in enumerate(row):
            if j > i:
                if G[i, j] == 1 and i not in changed and j not in changed:
                    maxIJ = max(G[i, i], G[j, j])
                    if G[i, i] != maxIJ:
                        changed.append(i)
                    else:
                        changed.append(j)
                    G[i, i] = maxIJ
                    G[j, j] = maxIJ
    # print(changed)


def checkAllEqual(G):
    return reduce((lambda x, y: x == y), np.diag(G))

queue = []
iterations = 100000
for i in range(iterations):
    count = 0
    G = np.zeros((n, n))
    G = randomizeEdges(G, n, p)
    G = randomAccumulate(G, n, p)
    # print(G)
    while not checkAllEqual(G):
        G = randomizeEdges(G, n, p)
        oneStepPropagation(G)
        # print(G)
        count +=1
    queue.append(count)
    count = 0

average = sum(queue)/iterations
print("It takes on average " + str(average) + " iterations until the entire network has updated in value.")
# %%

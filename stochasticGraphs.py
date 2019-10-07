%%markdown

# Stochastic Processes in Graphs
###### Han
# %%
%%markdown

The purposes of this notebook is to study stochastic processes and algorithms on graphs.

We will study random graphs $G = (N, p)$ where $N$ is fixed number of vertices and $p$ represents the probability of an edge being included.

We will represent graphs as an adjacency matrix $A$ of $G$.

$A_{i,j} = 1$ if there is an interaction between node i and node j.

$A_{i,i}$ represents the value stored by node i.

## Information Propagation in a Stochastic Network from Single Source

We start with studying the speed of information propagation in a graph where values of nodes are updated by the rule:

$A_{i,j} = 1 \implies \max{(A_{i,i},A_{j,j})} \rightarrow A_{i,i}, A_{j,j}$

After an adjacent node is updated, those nodes do not update adjacent nodes until the next time step or iteration of the propagation rule.

### Constant information

In this section we study how quickly constant information $C$ propagates in a constant sized graph $G = (N, p)$

i.e. how many timesteps or iterations of the update rule it takes before $A_{i, i} = A_{j, j} \; \forall \, i,j \in E(G)$

# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import pandas as pd


# graph initialization

n = 10
p = 0.5

def initializeGraph(n, p):
    upperDiagIndices = np.triu_indices(n, 1)
    A = np.zeros((n, n))
    return upperDiagIndices, A

def randomizeEdges(A, n, p):
    m = map((lambda x: x < p), np.random.rand(int((n-1) * n / 2)))
    edges = np.array([1 if e else 0 for e in m])
    A[upperDiagIndices] = edges
    return A


def randomAccumulate(A, n, p):
    randIdx = np.random.randint(low = 0, high = n)
    A[randIdx, randIdx] += 1
    return A


def drawGraph(A):
    nxA = nx.from_numpy_matrix(A)
    labelDict = dict(zip(nxA, np.diag(A).astype(int).tolist()))
    nx.draw(nxA, labels = labelDict)
    ax = plt.gca()
    ax.set_aspect('equal')


def oneStepPropagation(A):
    changed = []
    for i, row in enumerate(A):
        for j, column in enumerate(row):
            if j > i:
                if A[i, j] == 1 and i not in changed and j not in changed:
                    maxIJ = max(A[i, i], A[j, j])
                    if A[i, i] != maxIJ:
                        changed.append(i)
                    else:
                        changed.append(j)
                    A[i, i] = maxIJ
                    A[j, j] = maxIJ
    return A
    # print(changed)


def checkAllEqual(A):
    return (np.diag(A) == A[0,0]).all()

queue = []
iterations = 100000
for i in range(iterations):
    count = 0
    upperDiagIndices, A = initializeGraph(n, p)
    A = randomizeEdges(A, n, p)
    A = randomAccumulate(A, n, p)
    # print(A)
    while not checkAllEqual(A):
        A = randomizeEdges(A, n, p)
        A = oneStepPropagation(A)
        # print(A)
        count +=1
    queue.append(count)
    count = 0

average = sum(queue)/iterations
print("It takes on average " + str(average) + " iterations until the entire network has updated in value.")
# %%
%matplotlib inline
sns.countplot(list(map(int, queue)))
# %%
%%markdown
### Incrementing information

Here we introduce a new concept of incrementing information:

i.e. $C_t \geq C_{t-1} \; \forall t$

Where $t$ is time and $C_t$ is determined by an arbitrary monotonically increasing function $f(t)$

We measure the lag $L$ by:
$$L(G_t) = \sum_{i}^{N}{C_t - A_{i,i}}$$

For our simulation, we select $f(t) = t$
#%%

def randomAssign(A, n, p, t):
    randIdx = np.random.randint(low = 0, high = n)
    A[randIdx, randIdx] = t
    return A

def lag(A, t):
    return sum(np.ones(np.diag(A).shape) * t - np.diag(A))


def f(timestep):
    return timestep

maxTime = 50
iterations = 1000
n = 10
p = 1
lags = np.zeros((iterations, maxTime))
for iter in range(iterations):
    count = 0
    upperDiagIndices, A = initializeGraph(n, p)
    A = randomizeEdges(A, n, p)
    # print(A)
    while count < maxTime:
        A = oneStepPropagation(A)
        A = randomizeEdges(A, n, p)
        A = randomAssign(A, n, p, count)
        lags[iter, count] = lag(A, count)
        count += 1
    count = 0

# %%

%matplotlib inline
sns.lineplot(data=pd.DataFrame(lags).mean())
#
# average = sum(queue)/iterations
# print("It takes on average " + str(average) + " iterations until the entire network has updated in value.")
# %%

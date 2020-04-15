# importing libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

# importing dataset
df = pd.read_csv('AllBooks_baseline_DTM_Labelled_modified.csv')

tfidf_transformer = TfidfTransformer(norm = 'l2', use_idf = True)
tf_idf_vector = tfidf_transformer.fit_transform(df)

mat = tf_idf_vector.toarray()

# saving the TF-IDF Matrix
np.save('matrix.npy', mat)

# define cluster
class cluster:
    clus = {}
    def __init__(self,arg=[]):
        self.clus=set(arg)
        
# function to merge two clusters
def union(a, b):
	clubbed = cluster()
	clubbed.clus = a.clus | b.clus
	return clubbed

# function to determine two closest clusters based on chosen similarity measure
def get_min_pair(mat, A, sim_mat):
    temp = list(A)
    g1, g2 = -1, -1
    d = -1
    for i in range(len(temp)):
        for j in range(i):
            if(i != j):
                dist = min([sim_mat[m][n] for m in temp[i].clus for n in temp[j].clus])
                if (d == -1):
                    g1, g2 = i, j
                    d = dist
                elif (d > dist):
                    g1, g2 = i, j
                    d = dist
    return temp[g1], temp[g2]

# function to return proximity based on chosen proximity measure
def proximity(mat, i, j):
    return np.exp(-(mat[i].dot(mat[j].T)))

def modify_A(A, g1, g2):
    A = A - {g1}
    A = A - {g2}
    A.add(union(g1, g2))
    return A

# function to get data points corresponding to the 8 clusters of data
def agglo_HC(mat):
    l = len(mat)
    A = ([cluster([i])  for i in range(l)])
    sim_mat = np.zeros((l, l))
    for i in range(l):
        for j in range(i):
            sim_mat[i][j] = sim_mat[j][i] = proximity(mat, i, j)
    A = set(A)
    while 1:
        if len(A) > 8:
            g1, g2 = get_min_pair(mat, A, sim_mat)
            A = modify_A(A, g1, g2)
        else:
            break
    return A

A = agglo_HC(mat)

# saving the results into agglomerative.txt
results = open("agglomerative.txt",'w')
A = [sorted(list(i.clus)) for i in A]
A = sorted(A)
for i in A:
	print(*sorted(list(i)), sep = ",", file = results)
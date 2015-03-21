import numpy as np
#from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import scipy.linalg as la
import sys

np.set_printoptions(precision=4, suppress=True, edgeitems=8)

def load_file():
  cr1 = []
  with open('CR1') as f:
    for line in f:
      cr1.append(map(int, line.split()))
  return np.array(cr1)

def saruno_lda(data, n_size, ro):
  ends = np.cumsum(n_size)
  n_klas = np.size(n_size, 0)
  dim = np.size(data, 1)
  n1_size = np.hstack([0, n_size[0:n_klas-1]])
  m = []
  sw = np.zeros([dim, dim])

  for i in range(n_klas):
     m.append(np.mean(data[n1_size[i]: ends[i], :], axis=0))
     sw = sw + np.dot(n_size[i], np.cov(data.transpose()[:, n1_size[i]: ends[i]]))
  sw = sw / ends[-1]
  sw = np.dot(sw, (1-ro)) + np.diag(np.dot(np.diag(sw), ro))
  sb = np.cov(np.array(m).transpose())

  J_eig_val, v = np.linalg.eig(np.linalg.solve(sw, sb))  

  dd = np.diag(J_eig_val)
  ind = np.argsort(dd, axis=0)
  s = np.sort(dd, axis=0)

  s = s[::-1] # reverse
  ind = ind[::-1] # reverse

  W_LDA = []
  for i in range(n_klas - 1): 
    W_LDA.append(v[ind[i]]) # Largest (C-1) eigen vectors of matrix J
  return np.array(W_LDA).transpose()

def svd():
  cr1 = load_file()
  cm = np.cov(cr1.transpose())
  #u, s, vt = svds(cm)
  u, s, vt = np.linalg.svd(cm)
  #cr2 = cr1
  cr2 = np.dot(cr1, u[:, 0:3]) 
  plt.figure(1)
  plt.plot(cr2[0:500, 0],
           cr2[0:500, 1], 'k.',
           cr2[500:1000, 0],
           cr2[500:1000, 1], 'b.',
           cr2[4000:4500, 0],
           cr2[4000:4500, 1], 'r.')
  plt.show()

def main():
  #svd()
  
  cr1 = load_file()
  data = np.vstack([cr1[0:500, :], cr1[500:1000, :], cr1[4000:4500, :]])
  n_size = [500, 500, 500]
  ro = 0.00001

  W_LDA = saruno_lda(data, n_size, ro)

  cr2 = np.dot(data, W_LDA)
  plt.figure(2)
  plt.plot(cr2[0:500, 0],
           cr2[0:500, 1], 'k.',
           cr2[500:1000, 0],
           cr2[500:1000, 1], 'b.',
           cr2[1000:1500, 0],
           cr2[1000:1500, 1], 'r.')
  plt.show()

if __name__ == '__main__':
  main()

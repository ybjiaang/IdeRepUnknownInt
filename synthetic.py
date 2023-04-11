import numpy as np


class BipGraph():
  """
  The class that contains infomation about the bipartite causal graph between 
  observed and hidden variables
  """
  def __init__(self, num_hidden, num_observed, graphtype='purechild'):
    self.num_hidden = num_hidden
    self.num_observed = num_observed
    self.hidden_dom_size = np.zeros(num_hidden)
    self.adj = np.zeros((self.num_observed, self.num_hidden))
    self.graphtype = graphtype

  def gen_random_graph(self, prob = 0.5, attempts = 1000):
    if self.graphtype == 'purechild':
      if self.num_hidden >= self.num_observed:
        raise ValueError('num of hidden is not smaller than the num of observed')
      
      # generate a random biparite graph first
      self.adj = np.random.binomial(1, prob, size = self.adj.shape)
      
    
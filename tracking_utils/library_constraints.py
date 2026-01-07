from motile.constraints import Constraint
from motile.variables import NodeAppear


class Cardinality(Constraint):

  def __init__(self, num_tracks):
      self.num_tracks = num_tracks

  def instantiate(self, solver):
      appear_indicators = solver.get_variables(NodeAppear)
      yield sum([appear_indicators[n] for n in solver.graph.nodes]) == self.num_tracks




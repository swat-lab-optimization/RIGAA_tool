from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
#from pymoo.algorithms.moo.moead import MOEAD
#from pymoo.algorithms.soo.nonconvex.es import ES
#from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA

ALRGORITHMS = {
    "ga": GA,
    "nsga2": NSGA2,
    "smsemoa": SMSEMOA,
    "rigaa": NSGA2,
    "rigaa_s" : SMSEMOA,
    "random": RandomSearch,
    "random_rl": RandomSearch
}


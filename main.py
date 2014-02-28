__author__ = 'Wilbert'
import random
import sys
from datetime import datetime
import scipy.integrate
import numpy


class Member(object):
    def __init__(self, vector, dy, y0, t, expected, fitness=None):
        self.Vector = vector
        self.Expected = expected
        self.DY = dy
        self.Y0 = y0
        self.T = t
        self.Fitness = fitness if fitness is not None else self.GetFitness()
        return

    def GetFitness(self):
        obs = scipy.integrate.odeint(self.DY, self.Y0, self.T, args=(self.Vector,), mxstep=1000)
        fitness = []
        for i in range(len(obs)):
            fitness.extend([(obs[i][j] - self.Expected[i][j+1]) ** 2 for j in range(len(obs[i]))])
        return sum(fitness)

def dY(y, t, p):
    dy0 = p[0] - y[0] * 1. - (p[1] * y[0] - 0. * y[1]) * (1. + p[4] * y[1] ** 2.2)
    dy1 = (p[1] * y[0] - 0. * y[1]) * (1. + p[4] * y[1] ** 2.2) - y[1] * p[6]
    return [dy0, dy1]

def CreateRandomMember(vector_length, min_value, max_value, dY, y0, t):
    vector = [round(random.uniform(min_value, max_value), 7) for i in range(vector_length)]
    return Member(vector=vector, dy=dY, y0=y0, t=t, expected=data)

def CreateTrialValue(original, samples, CR=0.6, F=0.8):
    o = original.Vector
    a = samples[0].Vector
    b = samples[1].Vector
    c = samples[2].Vector

    value = []
    for i in range(len(o)):
        if random.random() <= CR:
            v = round(a[i] + F * (b[i] - c[i]), 7)
            if v > 0:
                value.append(v)
            else:
                value.append(0.)
        else:
            value.append(o[i])
    return Member(vector=value, dy=original.DY, y0=original.Y0, t=original.T, expected=original.Expected)



# Read file
file = open('expdata.txt','r')
data = []
for line in file:
    d = line.replace('\n','').split('  ')
    d = [float(x) for x in d]
    data.append(d)

# Integration conditions
y0 = [1., 0.]
t = numpy.linspace(0., 10., 50)

# Differential evolution
STOP = False
GENERATION_COUNT = 0
MAX_GENERATIONS = 10
FITNESS_THRESHOLD = 1e-6
PARAMETER_COUNT = 7

NP = PARAMETER_COUNT * 10 if PARAMETER_COUNT * 10 < 45 else 45
NI = 3      # Number of islands
MF = 0.45   # Migration frequency
NM = 1      # Number of migrants
SP = 3      # Selection policy = Randomly choose one of the top 3 for migration.
RP = 3      # Replacement policy = Randomly choose one of the top 3 for migration.
MT = range(NI)[1:] + [0]    # Migration topology: Ciricular, unidirectional

assert NP/NI > 3
if NI > 1:
    assert len(MT) == NI
NP = NP/NI * NI     # NP should be an exact multiple of NI.

lb = []
ub = []
for i in xrange(NP):
    if i % (NP/NI) == 0:
        lb.append(i)
    elif i % (NP/NI) == NP/NI - 1:
        ub.append(i)
island_boundaries = zip(lb, ub)


print('Starting DE search.')
clock = datetime.now()

POPULATION = []
for i in range(NP):
    POPULATION.append(CreateRandomMember(PARAMETER_COUNT, 0., 2., dY=dY, y0=y0, t=t))


while True:
    GENERATION_COUNT += 1
    samples = [random.sample(POPULATION, k=3) for i in range(NP)]
    trial_values = [CreateTrialValue(original=POPULATION[i], samples=samples[i]) for i in range(NP)]

    for i in range(NP):
        if trial_values[i].Fitness < POPULATION[i].Fitness:
            POPULATION[i] = trial_values[i]

    ISLANDS = []
    for i in range(NI):
        ISLANDS.append(sorted(POPULATION[island_boundaries[i][0]:island_boundaries[i][1]], key=lambda o: o.Fitness))
    top_members = [x[0] for x in ISLANDS]

    if GENERATION_COUNT >= MAX_GENERATIONS or min([x.Fitness for x in top_members]) < FITNESS_THRESHOLD:
        for member in top_members:
            print('Fitness: {0} Vector: {1}'.format(round(member.Fitness,3), member.Vector))
        break

print('Done optimizing.')
print('Optimization time: {0}'.format(datetime.now()-clock))



best = sorted(top_members, key=lambda o: o.Fitness)[0]
result = scipy.integrate.odeint(dY, t=t, y0=y0, args=(best.Vector,))
for i in range(len(result)):
    L = [t[i]]
    L.extend([x for x in result[i]])
    print(L)


print('Done.')








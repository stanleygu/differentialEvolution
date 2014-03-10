__author__ = 'Wilbert'
import random
import sys
from datetime import datetime
import scipy.integrate
import numpy
import matplotlib.pyplot as mplot


class DiffEvo(object):
    def __init__(self, dy, y0, t, expected):
        self.DY = dy
        self.Y0 = y0
        self.T = t
        self.Expected = expected
        self.Islands = []
        return

    def CreateIsland(self, population_size, vector_length, min_val, max_val):
        island = []
        for i in range(population_size):
            island.append(self.CreateRandomMember(vector_length, min_val, max_val))
        self.Islands.append(island)
        return

    def CreateRandomMember(self, vector_length, min_val, max_val):
        v = []
        for i in range(vector_length):
            v.append(round(random.uniform(min_val, max_val), 6))
        f = self.GetFitness(v)
        return Member(v,f)

    def CreateTrialMember(self, original, samples, CR=0.6, F=0.8):
        o = original.Vector
        a = samples[0].Vector
        b = samples[1].Vector
        c = samples[2].Vector

        new_vector = []
        for i in range(len(o)):
            if random.random() <= CR:
                v = round(a[i] + F * (b[i] - c[i]), 7)
                if v>0:
                    new_vector.append(v)
                else:
                    new_vector.append(o[i]/2.)
            else:
                new_vector.append(o[i])
        new_fitness = self.GetFitness(new_vector)
        return Member(new_vector, new_fitness)

    def GetFitness(self, vector):
        obs = scipy.integrate.odeint(self.DY, self.Y0, self.T, args=(vector,), mxstep=1000)
        sum_of_squares = 0.
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                if self.Expected[i][j+1] == 0:
                    0.
                else:
                    sum_of_squares += ((obs[i][j] - self.Expected[i][j+1])/ self.Expected[i][j+1]) ** 2
        return sum_of_squares

    def Migrate(self, probability, topology, selection_group, replacement_group):
        for i in range(len(self.Islands)):
            if random.random() <= probability:
                selection = random.choice(self.Islands[i][:selection_group])
                replacement_index = len(self.Islands[topology[i]]) - random.randint(1, replacement_group)
                self.Islands[topology[i]][replacement_index] = selection
        return

    def SortIslandsByFitness(self):
        for i in range(len(self.Islands)):
            self.Islands[i] = sorted(self.Islands[i], key=lambda o: o.Fitness)
        return

class Member(object):
    def __init__(self, vector, fitness):
        self.Vector = vector
        self.Fitness = fitness
        return


def dY(y, t, p):
    dy0 = p[0] - 1. * y[0] - (p[1] * y[0]) * (1. + p[2] * y[1] ** 2.2)
    dy1 = (p[1] * y[0]) * (1. + p[2] * y[1] ** 2.2) - y[1] * p[3]
    return [dy0, dy1]

def PlotResults(de):
    best_members = []
    solutions = []
    for i in range(len(de.Islands)):
        best_members.append(de.Islands[i][0])
        solutions.append(scipy.integrate.odeint(de.DY, de.Y0, de.T, args=(best_members[i].Vector, )))

    exp_t = [de.Expected[i][0] for i in range(len(de.Expected))]

    exp_y0 = [de.Expected[i][1] for i in range(len(de.Expected))]
    exp_y1 = [de.Expected[i][2] for i in range(len(de.Expected))]
    mplot.plot(exp_t, exp_y0, 'ko')
    mplot.plot(exp_t, exp_y1, 'ko')

    for i in range(len(best_members)):
        obs_y0 = solutions[i][:,0]
        obs_y1 = solutions[i][:,1]
        mplot.plot(de.T, obs_y0)
        mplot.plot(de.T, obs_y1)

    mplot.xlabel('Time')
    mplot.ylabel('Unknown')
    mplot.show()
    return



# Read file
file = open('expdata.txt','r')
data = []
for line in file:
    d = line.replace('\n','').split('  ')
    d = [float(x) for x in d]
    data.append(d)


# Differential evolution
GENERATION_COUNT = 0
MAX_GENERATIONS = 200
FITNESS_THRESHOLD = 1e-6
PARAMETER_COUNT = 4
NUMBER_OF_ISLANDS = 3      # Number of islands
MIGRATION_TOPOLOGY = range(NUMBER_OF_ISLANDS)[1:] + [0]    # Migration topology: Ring



print('Starting DE search.')
clock = datetime.now()

# Initialize Diff Evo routine
DE = DiffEvo(dy=dY, y0=[1., 0.], t=numpy.linspace(0., 10., 50), expected=data)

# Create islands
for i in range(NUMBER_OF_ISLANDS):
    DE.CreateIsland(20, PARAMETER_COUNT, 0., 10.)
    assert len(DE.Islands[i]) > 3


while True:
    GENERATION_COUNT += 1

    for island in DE.Islands:
        trial_values = []
        for i in range(len(island)):
            new_sample = random.sample(island, k=3)
            trial_values.append(DE.CreateTrialMember(island[i], new_sample))

        for i in range(len(island)):
            if trial_values[i].Fitness < island[i].Fitness:
                island[i] = trial_values[i]

        DE.SortIslandsByFitness()

    DE.Migrate(0.3, MIGRATION_TOPOLOGY, 3, 3)

    # Termination conditions
    best_member = []
    best_fitness = []
    for i in range(len(DE.Islands)):
        best_member.append(DE.Islands[i][0])
        best_fitness.append(DE.Islands[i][0].Fitness)

    if GENERATION_COUNT >= MAX_GENERATIONS or min(best_fitness) < FITNESS_THRESHOLD:
        for member in best_member:
            print('Fitness: {0} Vector: {1}'.format(round(member.Fitness,3), member.Vector))
        break

print('Done optimizing.')
print('Optimization time: {0}'.format(datetime.now()-clock))


PlotResults(de=DE)



print('Done.')








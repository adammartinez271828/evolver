"""Classes for objects that can sexually reproduce in the context of genetic search algorithms."""
from random import randint


class Evolver:
    """An abstact class for a creature that can sexually reproduce."""
    def reproduce_with(self, other):
        raise NotImplementedError('Evolver is an abstract class.')

    def mutate(self):
        raise NotImplementedError('Evolver is an abstract class.')


class Prokaryote(Evolver):
    """A creature that has only one chromosome."""
    def __init__(self, chromosome):
        if not isinstance(chromosome, list):
            raise ValueError('List required for chromosome argument,' +
                             ' received {} instead.'.format(type(chromosome)))
        self.chromosome = chromosome

    def reproduce_with(self, other):
        """Sexually reproduce with other via the crossover method.

        Randomly selects a split point k in the chromosome and returns a child such that given
             self's chromosome S = s_1, s_2, ... s_m and
            other's chromosome O = o_1, o_2, ... o_m then
            child's chromosome C = o_1, o_2, ... o_k, s_k+1, ... s_m

        Args:
            other: another Prokaryote
        """
        split_point = randint(1, len(self.chromosome)-1)
        return Prokaryote(self.chromosome[split_point:] + other.chromosome[:split_point])

    def mutate(self):
        raise NotImplementedError('Todo.')

    def __repr__(self):
        return '<Prokaryote: (' + ','.join(str(gene) for gene in self.chromosome) + ')>'

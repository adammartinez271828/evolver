"""Classes for objects that can sexually reproduce in the context of genetic search algorithms."""
import logging
from random import randint, sample

from scipy.stats import binom

log = logging.getLogger(__name__)


class Species:
    """An abstact class for a creature that can sexually reproduce."""
    def reproduce_with(self, other):
        """Abstract reproduction function.

        Args:
            other: an instance of type(self).

        Returns:
            a new subclass of Species with a genome derived from self and other.
        """
        raise NotImplementedError('Species is an abstract class.  Use a subclass of Species.')

    def mutate(self, mutation_func):
        """Abstract mutation function.

        Must modify self.
        Must guarantee that at least one mutation occurs.

        Args:
            mutation_func: function that is called to generate a new value for a mutated gene.
        """
        raise NotImplementedError('Species is an abstract class.  Use a subclass of Species.')


class Prokaryote(Species):
    """A creature that has only one chromosome."""
    def __init__(self, chromosome):
        """Create a Prokaryote.

        Args:
            chromosome: a single chromosome for this creature.  Must be of <type: list>.
        """
        if not isinstance(chromosome, list):
            raise ValueError('List required for chromosome, ' +
                             'received {} instead.'.format(type(chromosome)))
        self.chromosome = chromosome

    def reproduce_with(self, other):
        """Sexually reproduce with other via the crossover method.

        Randomly selects a split point k in the chromosome and returns a child such that given
             self's chromosome S = s_1, s_2, ... s_m and
            other's chromosome O = o_1, o_2, ... o_m then
            child's chromosome C = s_k+1, ... s_m, o_1, ... o_k

        Args:
            other: another Prokaryote
        """
        split_point = randint(1, len(self.chromosome)-1)
        return Prokaryote(self.chromosome[split_point:] + other.chromosome[:split_point])

    def mutate(self, mutation_func):
        """Use a bit-string mutation approach to mutating the chromosome.

        For a chromosome of length n genes, each gene has a 1/n probability of being mutated.
        Guarantees at least one mutation occurs.  Mutates self.  Utilizes the binomial distribution
        and random sampling IOT minimize C-style looping.

        Args:
            mutation_func: function to use to mutate the selected gene.  Must take a gene as input.
        """
        num_mutations = 0
        while num_mutations == 0:  # Guarantee at least one mutation occurs.
            num_mutations = binom.rvs(n=len(self.chromosome), p=1.0/len(self.chromosome))
        # log.debug('Mutating %d genes in a Prokaryote.', num_mutations)

        # Select genes without replacement and mutate them.
        genes_with_indexes = list(enumerate(self.chromosome))
        mutating_genes = sample(genes_with_indexes, num_mutations)
        for gene_index, gene in mutating_genes:
            self.chromosome[gene_index] = mutation_func(gene)

    def __repr__(self):
        return '<Prokaryote: (' + ','.join(str(gene) for gene in self.chromosome) + ')>'

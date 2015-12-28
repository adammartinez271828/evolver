"""Module to encapsulate various genetic algorithms.  There is a wide variety of genetic algorithms
and this is intended to be a dropzone for different ones that can be imported into other
projects."""
from copy import deepcopy
import logging
from random import sample, shuffle

from scipy.stats import binom

log = logging.getLogger(__name__)


def truncation_selection(population, fit_func):
    """Breed the next generation using a truncation selection algorithm.  The most fit half of the
    population will survive to breeding age.  Each survivor will choose a spouse that is not
    themselves at random and produce two children.

    Args:
        population: a list of Evolvers
        fit_func: the fitness function used to determine survivors from generation to generation

    Returns:
        the next generation of Evolvers
    """
    num_survivors = len(population)//2
    survivors = sorted(population, key=fit_func)[:num_survivors]
    spouses = deranged(survivors)  # guarantees no survivor will be paired with itself.

    # Reversing the order for seconds helps with crossover reproduction methods.
    firsts = [survivor.reproduce_with(spouse) for survivor, spouse in zip(survivors, spouses)]
    seconds = [spouse.reproduce_with(survivor) for survivor, spouse in zip(survivors, spouses)]

    new_population = firsts + seconds

    return new_population


def truncation_with_mutation(population, fit_func, mutation_rate, mutation_func):
    """Breed the next generation using a truncation selection algorithm with mutation.  The most fit
    half of the population will survive to breeding age.  Each survivor will choose a spouse that is
    not themselves at random and produce two children.  Each child will have mutation_rate chance
    to mutate.

    Args:
        population: a list of Evolvers
        fit_func: the fitness function used to determine survivors from generation to generation
        mutation_rate: chance for each reproduction event to trigger a mutation event
        mutation_func: the function responsible for controlling mutation.  Takes one argument of:
            a gene, a chromosome, or a genotype, depending on the context of the mutation.

    Returns:
        the next generation of Evolvers
    """
    new_population = truncation_selection(population, fit_func)

    # Given a population of n creatures each with a p chance of mutation, the number of
    # mutations in a given population obeys a binomial distribution.  i.e. given a population
    # of 100 creatures each with a 1% chance of mutation, you would expect to see 1 mutation
    # per generation on average, but there is a ~36.6% chance 0 will mutate, a ~37% chance 1
    # will mutate, an ~18.5% chance 2 will mutate, etc.
    num_mutations = binom.rvs(n=len(new_population), p=mutation_rate)
    if num_mutations > 0:
        # log.debug('Generating %d mutations in %d creatures.', num_mutations, len(new_population))
        mutators = sample(new_population, num_mutations)
        for mutator in mutators:
            mutator.mutate(mutation_func)

    return new_population


def deranged(iterable):
    """Generates a mathematical derangement of iterable.  i.e. take a sequence of items and return
    a sequence of those same items in random order such that the nth item of the original sequence
    and the nth item of the deranged sequence are never the same item.

    This is actually implemented by just repeatedly calling random.shuffle on a copy of the iterable
    until a derangement is found.  On average, random.shuffle will be called e (2.718...) times.

    Args:
        iterable: an iterable object.

    Returns
        a deranged list of items in iterable
    """
    deranged_copy = deepcopy(iterable)
    while True:
        shuffle(deranged_copy)
        matches = [a == b for a, b in zip(iterable, deranged_copy)]
        if not any(matches):
            break

    return deranged_copy

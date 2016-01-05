"""Module to encapsulate various genetic algorithms.  There is a wide variety of genetic algorithms
and this is intended to be a dropzone for different ones that can be imported into other
projects."""
import logging
from random import choice, random, sample, shuffle

from scipy.stats import binom

log = logging.getLogger(__name__)


def truncation_with_mutation(population, fit_func, mutation_rate, mutation_func, ratio=2):
    """Breed the next generation using a truncation selection algorithm with mutation.

    The most fit half of the population will survive to breeding age.  Each survivor will choose a
    spouse that is not themselves at random and produce two children.  Each child will have
    mutation_rate chance to mutate.

    Args:
        population: a list of Evolvers
        fit_func: the fitness function used to determine survivors from generation to generation
        mutation_rate: probability of each reproduction event triggering a mutation event
        mutation_func: the function responsible for controlling mutation.
        ratio: proportion of generation that survives to breeding age, integer >= 2, optional,
            default 2

    Returns:
        the next generation of Evolvers
    """
    new_population = truncation_selection(population, fit_func, ratio=ratio)

    mutate(new_population, mutation_rate, mutation_func)

    return new_population


def fitness_proportional_selection(population, fit_func):
    """Breed the next generation using fitness proportional selection.

    Each Evolver's probability of being selected for a reproduction event is proportional to their
    fitness.  Given an Evolver E with fitness fe and n other Evolvers with fitnesses f1, f2 ... fn,
    then E's chance of reproducing is:
        fe / (f1 + f2 + ... + fn + fe)
    Note that this does not work when minimizing fitness.

    Args:
        population: a list of Evolvers
        fit_func: the fitness function used to determine survivors from generation to generation
    """
    # Going to implement this in the most straightforward way.
    # TODO: Fix inefficient looping in Fitness-Proportional Selection
    # Precompute fit_func since we are going to be calling it a lot.
    pop_with_fitness = [(fit_func(creature), creature) for creature in population]
    total_fitness = sum(fitness for fitness, creature in pop_with_fitness)
    # log.debug('Total pop fitness: %d', total_fitness)

    new_population = []
    for _ in range(len(population)):
        while True:
            fitness, father = choice(pop_with_fitness)
            selection_chance = fitness/total_fitness
            # log.debug('Father selection chance: %f', selection_chance)
            if random() < selection_chance:
                # log.debug('Father selected.')
                break
        while True:
            fitness, mother = choice(pop_with_fitness)
            selection_chance = fitness/total_fitness
            # log.debug('Mother selection chance: %f', selection_chance)
            if random() < selection_chance and mother is not father:
                # log.debug('Mother selected.')
                break
        new_population.append(father.reproduce_with(mother))

    return new_population


def truncation_selection(population, fit_func, ratio=2):
    """Breed the next generation using a truncation selection algorithm.

    The most fit 1/ratio of the population will survive to breeding age.  Each survivor will choose
    a spouse that is not themselves at random and produce ratio children.

    Args:
        population: a list of Evolvers
        fit_func: the fitness function used to determine survivors from generation to generation
        ratio: proportion of generation that survives to breeding age, integer >= 2, optional,
            default 2

    Returns:
        the next generation of Evolvers
    """
    num_survivors = len(population)//ratio
    survivors = sorted(population, key=fit_func, reverse=True)[:num_survivors]
    spouses = deranged(survivors)  # guarantees no survivor will be paired with itself.

    new_population = []
    for survivor, spouse in zip(survivors, spouses):
        for _ in range(ratio):
            new_population.append(survivor.reproduce_with(spouse))
    while len(new_population) < len(population):
        # In the case where population % ratio != 0, you may need to generate a few extra children.
        new_population.append(survivor.reproduce_with(spouse))

    return new_population


def mutate(population, mutation_rate, mutation_func):
    """Mutates each Evolver in population with fixed probability.

    Given a population of n creatures each with a p chance of mutation, the number of
    mutations in a given population obeys a binomial distribution.  i.e. given a population
    of 100 creatures each with a 1% chance of mutation, you would expect to see ~1 mutation
    per generation, but there is a ~36.6% chance 0 will mutate, a ~37% chance 1  will mutate, an
    ~18.5% chance 2 will mutate, etc.

    Args:
        population: a list of Evolvers
        mutation_rate: chance for each Evolver to mutate
        mutation_func: function that is applied to the Evolver's genome upon mutation
    """
    num_mutations = binom.rvs(n=len(population), p=mutation_rate)
    if num_mutations > 0:
        # log.debug('Generating %d mutations in %d creatures.', num_mutations, len(population))
        mutators = sample(population, num_mutations)
        for mutator in mutators:
            mutator.mutate(mutation_func)

    return population


def deranged(iterable):
    """Generates a mathematical derangement of iterable.

    A derangement is a permutation of a set such that no item remains in its original place.  This
    is actually implemented by just repeatedly calling random.shuffle on a copy of the iterable
    until a derangement is found.  On average, random.shuffle will be called e (2.718...) times.

    Args:
        iterable: an iterable object.

    Returns
        a deranged list of items in iterable
    """
    deranged_copy = iterable.copy()
    while True:
        shuffle(deranged_copy)
        matches = (a == b for a, b in zip(iterable, deranged_copy))
        if not any(matches):
            break

    return deranged_copy

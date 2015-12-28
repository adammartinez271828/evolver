"""Main module to execute a genetic algorithm to create creatures with genomes that sum to 170."""
from copy import deepcopy
import logging
from random import randint, shuffle
from statistics import mean

from evolvers import Prokaryote

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

MAX_GENERATIONS = 1000
POP_SIZE = 100
FITNESS_THRESHOLD = 1


def main():
    # Generate a pop_size list of prokaryotes with genomes consisting of 10 random integers
    population = [Prokaryote([randint(0, 50) for _ in range(10)]) for __ in range(POP_SIZE)]

    current_generation = 0

    while current_generation < MAX_GENERATIONS:
        # log.debug('Population size: %d on generation: %d', len(population), current_generation)
        current_generation += 1
        population = truncate_algorithm(population)
        average_fitness = mean([calculate_fitness(creature) for creature in population])
        if average_fitness < FITNESS_THRESHOLD:
            log.debug('Breaking on generation %d, average fitness is %f.', current_generation,
                      average_fitness)
            break
    else:
        log.debug('Exceeding %d generations, average fitness is %f.', MAX_GENERATIONS,
                  average_fitness)

    return population


def truncate_algorithm(population):
    """Breed the next generation using a truncation selection algorithm.  The most fit half of the
    population will survive to breeding age.  Each survivor will choose a spouse that is not
    themselves at random and produce two children.

    Args:
        population: a list of Evolvers

    Returns:
        the next generation of Evolovers
    """
    num_survivors = len(population)//2
    survivors = sorted(population, key=calculate_fitness)[:num_survivors]
    spouses = deranged(survivors)  # guarantees no survivor will be paired with itself.

    # Reversing the order for seconds helps with crossover reproduction methods.
    firsts = [survivor.reproduce_with(spouse) for survivor, spouse in zip(survivors, spouses)]
    seconds = [spouse.reproduce_with(survivor) for survivor, spouse in zip(survivors, spouses)]

    return firsts + seconds


def calculate_fitness(creature):
    """Calculates the fitness of a creature.  A lower score is better.

    Args:
        creature: a Prokaryote

    Returns:
        fitness score: a positive float representing the fitness of the creature.  0 is perfect
            fitness.
    """
    score = sum(creature.chromosome)
    return abs(170 - score)


def deranged(iterable):
    """Generates a mathematical derangement of iterable.  In layman's terms, it takes a sequence of
    numbers and returns a sequence of those same numbers in a random order such that the nth item
    of the original sequence and the nth item of the deranged sequence are never the same item.
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


if __name__ == '__main__':
    results = main()
    # log.debug('Dumping current population and fitnesses:')
    # results = [(creature, calculate_fitness(creature))
    #              for creature in sorted(results, key=calculate_fitness)]
    # print('\n'.join(('{}: {}'.format(c, f) for c, f in results)))

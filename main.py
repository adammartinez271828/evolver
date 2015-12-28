"""Main module to execute a genetic algorithm to create creatures with genomes that sum to 170."""
import logging
from random import randint
from statistics import mean

# from evolution_algorithms import truncation_selection as next_gen
from evolution_algorithms import truncation_with_mutation as next_gen
from evolvers import Prokaryote


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

MAX_GENERATIONS = 1000
POP_SIZE = 100
FITNESS_THRESHOLD = 1
MUTATION_RATE = 0.075


def main():
    # Generate a population of Prokaryotes with genomes consisting of 10 random integers 0 to 50
    population = [Prokaryote([randint(0, 50) for _ in range(10)]) for __ in range(POP_SIZE)]

    current_generation = 0

    while current_generation < MAX_GENERATIONS:
        # log.debug('Population size: %d on generation: %d', len(population), current_generation)
        current_generation += 1
        population = next_gen(population,
                              fit_func=calculate_fitness,
                              mutation_rate=MUTATION_RATE,
                              mutation_func=lambda gene: randint(0, 50)
                              )
        mean_fitness = mean([calculate_fitness(creature) for creature in population])
        if mean_fitness < FITNESS_THRESHOLD:
            log.debug('Breaking on generation %d, mean fitness is %f.', current_generation,
                      mean_fitness)
            break
    else:
        log.debug('Exceeding %d generations, mean fitness is %f.', MAX_GENERATIONS, mean_fitness)

    return population


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


if __name__ == '__main__':
    results = main()
    # log.debug('Dumping current population and fitnesses:')
    # results = [(creature, calculate_fitness(creature))
    #              for creature in sorted(results, key=calculate_fitness)]
    # log.debug('\n    ' + '\n    '.join(('{}: {}'.format(c, f) for c, f in results)))

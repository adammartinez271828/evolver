"""Main module to execute a genetic algorithm to create creatures with genomes that sum to 170."""
import logging
from os import listdir, path, remove
from random import randint
from statistics import median

# from evolution_algorithms import truncation_selection as next_gen
# from evolution_algorithms import truncation_with_mutation as next_gen
from evolution_algorithms import fitness_proportional_selection as next_gen
from evolvers import Prokaryote


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

MAX_GENERATIONS = 1000
POP_SIZE = 100
TARGET_FITNESS = 0.95
MUTATION_RATE = 0.075
DUMP_POP = True
DUMP_INTERVAL = 100


def main():
    # Generate a population of Prokaryotes with genomes consisting of 10 random integers 0 to 50
    population = [Prokaryote([randint(0, 50) for _ in range(10)]) for __ in range(POP_SIZE)]

    current_generation = 0
    if DUMP_POP:
        for filename in listdir('dmp'):
            remove(path.join('dmp', filename))

    while current_generation < MAX_GENERATIONS:
        # log.debug('Population size: %d on generation: %d', len(population), current_generation)
        if DUMP_POP & (current_generation % DUMP_INTERVAL == 0):
            dump(get_dump_path(current_generation), population)
        current_generation += 1

        population = next_gen(population,
                              fit_func=calculate_fitness,
                              # mutation_rate=MUTATION_RATE,
                              # mutation_func=lambda gene: randint(0, 50),
                              # ratio=3
                              )
        median_fitness = median([calculate_fitness(creature) for creature in population])
        if median_fitness >= TARGET_FITNESS:
            log.debug('Breaking on generation %d, median fitness is %f.', current_generation,
                      median_fitness)
            break
    else:
        log.debug('Exceeding %d generations, median fitness is %f.', MAX_GENERATIONS,
                  median_fitness)

    return population


def calculate_fitness(creature):
    """Calculates the fitness of a creature.

    Ideally, this should be a function continuous on (0, 1] where higher scores represent
    better fitness.

    Args:
        creature: a Prokaryote

    Returns:
        a positive float <= 1 representing the fitness of the creature.
    """
    target_value = 170
    score = abs(target_value - sum(creature.chromosome))
    return target_value / (score + target_value)


def get_dump_path(current_generation):
    """Gets the path for a dumpfile at the current generation.

    Generates filename such that the operating system can sort them in lexicographical order.

    Args:
        current_generation: generation for dumpfile

    Returns:
        system-specific path to dump a dumpfile, i.e. 'dmp/dmp_*.txt' or 'dmp\dmp_*.txt'
    """
    num = str(current_generation).zfill(len(str(MAX_GENERATIONS)))
    return path.join('dmp', 'dmp_{}.txt'.format(num))


def dump(filename, population, mode='w'):
    """Dumps the population to a file for later analysis.

    Args:
        filename: name of file to dump to
        population: list of Evolvers
        mode: write mode, optional, defaults to 'w'
    """
    with open(filename, mode=mode) as fd:
        out = ((calculate_fitness(creature), creature) for creature in population)
        for score, creature in sorted(out, key=lambda x: x[0], reverse=True):
            fd.write('{}: {}'.format(score, creature) + '\n')


if __name__ == '__main__':
    results = main()
    log.debug('Dumping current population and fitnesses:')
    results = [(calculate_fitness(creature), creature)
               for creature in sorted(results, key=calculate_fitness, reverse=True)]
    log.debug('\n    ' + '\n    '.join(('{0:0.3f}: {1}'.format(fitness, creature)
                                        for fitness, creature in results)))

"""Permutation-based Genetic Algorithm for the Tourist Trip Design Problem (TTDP)."""

from dataclasses import dataclass
import random
import time
from typing import Sequence

import numpy as np

from .decoder import decode_chromosome
from .models import ItineraryResult, POI, TripConfig


@dataclass(frozen=True)
class GAConfig:
    """Hyperparameters for the Genetic Algorithm."""

    population_size: int = 80
    generations: int = 150
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    tournament_size: int = 3
    elitism_count: int = 2


@dataclass
class GAResult:
    """Output of a single Genetic Algorithm run."""

    seed: int
    best_chromosome: list[int]
    best_itinerary: ItineraryResult
    best_fitness_history: list[float]
    avg_fitness_history: list[float]
    runtime_sec: float


def _tournament_select(
    population: list[list[int]],
    fitnesses: list[float],
    tournament_size: int,
    rng: random.Random,
) -> list[int]:
    """Select one parent chromosome using tournament selection."""
    candidates = rng.sample(range(len(population)), k=tournament_size)
    best_idx = max(candidates, key=lambda idx: fitnesses[idx])
    return population[best_idx][:]


def _order_crossover(
    parent1: Sequence[int], parent2: Sequence[int], rng: random.Random
) -> tuple[list[int], list[int]]:
    """Perform Order Crossover (OX) on two permutation chromosomes."""
    size = len(parent1)
    if size < 2:
        return list(parent1), list(parent2)

    cut1, cut2 = sorted(rng.sample(range(size), 2))

    def _build_child(p_a: Sequence[int], p_b: Sequence[int]) -> list[int]:
        child = [-1] * size
        child[cut1 : cut2 + 1] = p_a[cut1 : cut2 + 1]
        in_slice = set(child[cut1 : cut2 + 1])
        remaining = [gene for gene in p_b if gene not in in_slice]

        fill_idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[fill_idx]
                fill_idx += 1
        return child

    return _build_child(parent1, parent2), _build_child(parent2, parent1)


def _mutate(chromosome: list[int], mutation_rate: float, rng: random.Random) -> list[int]:
    """Apply Swap or Inversion mutation with probability mutation_rate."""
    if rng.random() >= mutation_rate or len(chromosome) < 2:
        return chromosome

    mutated = chromosome[:]
    i, j = sorted(rng.sample(range(len(mutated)), 2))

    if rng.random() < 0.5:
        # Swap mutation
        mutated[i], mutated[j] = mutated[j], mutated[i]
    else:
        # Inversion mutation
        mutated[i : j + 1] = reversed(mutated[i : j + 1])

    return mutated


def run_ga(
    pois: Sequence[POI],
    dist_matrix: np.ndarray,
    time_matrix: np.ndarray,
    trip_config: TripConfig,
    ga_config: GAConfig = GAConfig(),
    seed: int = 42,
) -> GAResult:
    """Run permutation-based Genetic Algorithm for TTDP."""
    start_clock = time.perf_counter()
    rng = random.Random(seed)

    poi_ids = [poi.id for poi in pois if poi.id != 0]
    population: list[list[int]] = []
    for _ in range(ga_config.population_size):
        chrom = poi_ids[:]
        rng.shuffle(chrom)
        population.append(chrom)

    best_fitness_history: list[float] = []
    avg_fitness_history: list[float] = []

    best_overall_chrom: list[int] = []
    best_overall_result: ItineraryResult | None = None

    for _gen in range(ga_config.generations):
        decoded = [
            decode_chromosome(chrom, pois, dist_matrix, time_matrix, trip_config)
            for chrom in population
        ]
        fitnesses = [res.fitness for res in decoded]

        gen_best_idx = max(range(len(population)), key=lambda idx: fitnesses[idx])
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_avg_fitness = float(np.mean(fitnesses))

        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(gen_avg_fitness)

        if best_overall_result is None or gen_best_fitness > best_overall_result.fitness:
            best_overall_result = decoded[gen_best_idx]
            best_overall_chrom = population[gen_best_idx][:]

        # Elitism
        sorted_indices = sorted(
            range(len(population)), key=lambda idx: fitnesses[idx], reverse=True
        )
        next_population: list[list[int]] = [
            population[idx][:] for idx in sorted_indices[: ga_config.elitism_count]
        ]

        # Offspring generation
        while len(next_population) < ga_config.population_size:
            parent1 = _tournament_select(
                population, fitnesses, ga_config.tournament_size, rng
            )
            parent2 = _tournament_select(
                population, fitnesses, ga_config.tournament_size, rng
            )

            if rng.random() < ga_config.crossover_rate:
                child1, child2 = _order_crossover(parent1, parent2, rng)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = _mutate(child1, ga_config.mutation_rate, rng)
            next_population.append(child1)

            if len(next_population) < ga_config.population_size:
                child2 = _mutate(child2, ga_config.mutation_rate, rng)
                next_population.append(child2)

        population = next_population

    runtime_sec = time.perf_counter() - start_clock
    assert best_overall_result is not None

    return GAResult(
        seed=seed,
        best_chromosome=best_overall_chrom,
        best_itinerary=best_overall_result,
        best_fitness_history=best_fitness_history,
        avg_fitness_history=avg_fitness_history,
        runtime_sec=runtime_sec,
    )


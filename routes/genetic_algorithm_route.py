#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Load LaDe dataset (replace path with actual download)
# Dataset available at: https://github.com/wenhaomin/LaDe
try:
    df = pd.read_csv('datasets/delivery_sh.csv')
    print(f"Loaded {len(df)} delivery records from Shanghai")
except FileNotFoundError:
    print("Dataset not found! Using synthetic data instead")
    # Generate synthetic delivery points if dataset unavailable
    np.random.seed(42)
    delivery_points = np.random.uniform(low=[31.0, 121.0], high=[31.3, 121.5], size=(50,2))
    depot = np.array([31.15, 121.3])
else:
    # Preprocess actual data
    delivery_points = df[['lat', 'lng']].values
    depot = np.mean(delivery_points, axis=0)

# --- Summarized Evolutionary Hyperparameters ---

# NUM_POINTS: Defines the genetic makeup and complexity of the problem.
# - Genetic Analogy: The number of genes on a chromosome, where each gene is a delivery point defining the route.
# - Computational Impact: Directly sets the problem's search space; more points exponentially increase complexity.
NUM_POINTS = 50

# POPULATION_SIZE: The size of the evolving gene pool.
# - Genetic Analogy: The number of chromosomes (routes) in each generation, representing the population's genetic diversity.
# - Computational Impact: A larger population improves solution exploration but increases the computational load per generation.
POPULATION_SIZE = 100

# GENERATIONS: The duration of the evolutionary process.
# - Genetic Analogy: The number of evolutionary cycles for the population of routes to adapt and improve their fitness.
# - Computational Impact: More generations allow for better convergence toward an optimal route but increase total runtime.
GENERATIONS = 500

# MUTATION_RATE: The rate of spontaneous genetic change.
# - Genetic Analogy: The probability of a random gene (delivery point) swap, introducing novel traits to escape local optima.
# - Computational Impact: A low-cost operation crucial for maintaining diversity and preventing premature convergence.
MUTATION_RATE = 0.02

# TOURNAMENT_SIZE: The intensity of natural selection.
# - Genetic Analogy: The size of the "survival of the fittest" competition that determines which routes reproduce.
# - Computational Impact: A larger size increases selection pressure, which can speed up convergence but may reduce diversity.
TOURNAMENT_SIZE = 5

# ELITISM_COUNT: The mechanism for preserving elite traits.
# - Genetic Analogy: The number of elite chromosomes (best routes) whose superior genetic code is passed on unchanged.
# - Computational Impact: A computationally cheap way to ensure the best-found solution is never lost, accelerating progress.
ELITISM_COUNT = 2


# Create distance matrix
points = np.vstack([depot, delivery_points[:NUM_POINTS]])
distance_matrix = np.zeros((len(points), len(points)))
for i in range(len(points)):
    for j in range(len(points)):
        distance_matrix[i][j] = euclidean(points[i], points[j])

def create_route():
    """Generate random route excluding depot (index 0)"""
    return random.sample(range(1, len(points)), len(points)-1)

def route_distance(route):
    """Calculate total distance for a route (depot -> deliveries -> depot)"""
    total = distance_matrix[0][route[0]]  # Depot to first point
    for i in range(len(route)-1):
        total += distance_matrix[route[i]][route[i+1]]
    total += distance_matrix[route[-1]][0]  # Last point to depot
    return total

def tournament_selection(population, fitness):
    """Select parents using tournament selection"""
    tournament = random.sample(list(zip(population, fitness)), TOURNAMENT_SIZE)
    return min(tournament, key=lambda x: x[1])[0]

def ordered_crossover(parent1, parent2):
    """OX crossover for permutation representation"""
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [-1] * len(parent1)
    child[start:end+1] = parent1[start:end+1]
    
    # Fill remaining positions from parent2
    current_pos = 0
    for gene in parent2:
        if gene not in child:
            while current_pos < len(child) and child[current_pos] != -1:
                current_pos += 1
            if current_pos < len(child):
                child[current_pos] = gene
    return child

def swap_mutation(route):
    """Swap two random positions in the route"""
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Initialize population
population = [create_route() for _ in range(POPULATION_SIZE)]
best_fitness = float('inf')
best_route = None
history = []

for gen in range(GENERATIONS):
    # Evaluate fitness
    fitness = [route_distance(route) for route in population]
    
    # Track best solution
    current_best = min(fitness)
    if current_best < best_fitness:
        best_fitness = current_best
        best_route = population[fitness.index(current_best)]
    history.append(current_best)
    
    # Selection and reproduction
    new_population = []
    
    # Elitism: preserve best routes
    elite_indices = np.argsort(fitness)[:ELITISM_COUNT]
    new_population.extend([population[i] for i in elite_indices])
    
    # Create next generation
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population, fitness)
        parent2 = tournament_selection(population, fitness)
        child = ordered_crossover(parent1, parent2)
        child = swap_mutation(child)
        new_population.append(child)
    
    population = new_population

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history)
plt.title('Optimization Progress')
plt.xlabel('Generation')
plt.ylabel('Best Distance (meters)')

plt.subplot(1, 2, 2)
plt.scatter(points[1:,0], points[1:,1], c='blue', label='Delivery Points')
plt.scatter(*depot, c='red', s=100, marker='s', label='Depot')

# Plot best route
route_points = points[best_route]
plt.plot([depot[0], route_points[0][0]], [depot[1], route_points[0][1]], 'k--')
for i in range(len(route_points)-1):
    plt.plot([route_points[i][0], route_points[i+1][0]], 
             [route_points[i][1], route_points[i+1][1]], 'k--')
plt.plot([route_points[-1][0], depot[0]], [route_points[-1][1], depot[1]], 'k--')
plt.legend()
plt.title('Optimized Delivery Route')
plt.tight_layout()
plt.show()

print(f"Best route distance: {best_fitness:.2f} meters")






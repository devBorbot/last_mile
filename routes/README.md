# Ant Colony Optimization

Ant Colony Optimization (ACO) is a probabilistic technique used to solve computational problems that can be reduced to finding optimal paths through graphs. It's a member of the swarm intelligence family of algorithms and is inspired by the foraging behavior of real ants.

## How Ant Colony Optimization Works

The core idea of ACO is modeled on how ants find the shortest path between their colony and a food source.

1. **Exploration and Pheromone Trails**: Initially, ants wander randomly in search of food. When an ant finds a food source, it returns to the colony, laying down a chemical trail of pheromones along its path.
2. **Path Reinforcement**: Other ants, upon encountering this pheromone trail, are likely to follow it. As more ants travel this path to the food and back, they each deposit more pheromones, reinforcing the trail.
3. **Finding the Optimal Path**: Ants on shorter paths will complete their round trips faster than ants on longer paths. This means that in the same amount of time, the shorter path will be reinforced with more pheromones, making its scent stronger. This stronger trail attracts even more ants.
4. **Pheromone Evaporation**: The pheromone trails are not permanent; they evaporate over time. If a path is no longer used, or used infrequently (like a long, inefficient route), its pheromone trail will decay and disappear. This evaporation prevents the colony from getting stuck on a suboptimal path and allows for dynamic adaptation if the environment changes.

This combination of path reinforcement and pheromone evaporation allows the ant colony to collectively converge on the most efficient route.

## The ACO Algorithm Cycle

In a computational setting, the ACO algorithm simulates this behavior using "artificial ants" to find solutions to optimization problems like the Traveling Salesman Problem (TSP). The process generally follows these steps:

* **Initialization**: The algorithm starts by initializing parameters and placing a number of artificial ants at a starting node (the "colony"). Pheromone trails on all paths are set to an initial value.
* **Solution Construction**: Each ant builds a solution (e.g., a path visiting all cities in a TSP) by moving from node to node. The choice of the next node is probabilistic and is influenced by two main factors:
    * The strength of the pheromone trail on the path.
    * A heuristic value, such as the inverse of the distance to the next node (shorter distances are more attractive).
* **Pheromone Update**: After all ants have completed their tours, the pheromone trails are updated. This happens in two phases:
    * **Evaporation**: The pheromone level on all paths is reduced by a certain percentage to simulate natural evaporation.
    * **Reinforcement**: Pheromones are added to the paths that the ants traveled. Shorter, more optimal paths receive a greater amount of pheromones.
* **Termination**: This cycle of solution construction and pheromone updates is repeated for a set number of iterations or until a satisfactory solution has been found.

## How the Code Works

1. **Data Loading and Filtering**: The `load_and_filter_data` function reads the `delivery_sh.csv` file using pandas. It then filters the data to isolate the records for a single courier (`courier_id=2130`) on a single day (`ds=1010`). It extracts the latitude and longitude coordinates for these deliveries.
2. **Depot and Distance Matrix**: A central depot is calculated as the average of all delivery coordinates. This depot and the delivery locations are combined into one set of points. The `calculate_distance_matrix` function then computes the Euclidean distance between every pair of points, creating a matrix that the ACO algorithm will use.
3. **Running the ACO Algorithm**: The `run_aco` function orchestrates the entire optimization process. It initializes a pheromone matrix and then enters a loop for a specified number of iterations. In each iteration, it simulates a colony of ants building paths, and then it updates the pheromone trails based on the quality of those paths.
4. **Ant Path Construction**: For each ant, a path is built step-by-step. The `choose_next_node` function is the core of an ant's decision-making. It calculates the probability of moving to each unvisited node based on a combination of the pheromone level on the path and the distance to that node.
5. **Pheromone Update**: After all ants complete their tours, the `update_pheromones` function modifies the pheromone matrix. First, all trails are reduced by a `decay` factor to simulate evaporation. Then, pheromones are added to the paths the ants traveled, with shorter paths receiving a larger deposit. This reinforces good solutions.
6. **Termination and Result**: The algorithm repeats this process, and with each iteration, the pheromone trails on the shorter paths become stronger, guiding the ants toward an optimal or near-optimal solution. The function returns the best path found across all iterations.

## Explanation of Model Parameters

The effectiveness of the ACO algorithm is highly dependent on its parameters. Tuning them is key to finding good solutions efficiently.

* `n_ants` (Number of Ants): This determines how many potential solutions (paths) are explored in each iteration. A **higher number** of ants allows for a broader search of the solution space but increases computational cost. A **lower number** is faster but may lead to premature convergence on a suboptimal solution.
* `n_iterations` (Number of Iterations): This is the total number of cycles the algorithm will run. More iterations give the algorithm more time to refine the pheromone trails and converge on a better solution. However, too many iterations may be unnecessary if the solution stops improving.
* `decay` (Pheromone Evaporation Rate): This value, between 0 and 1, controls how quickly pheromone trails fade. A **high decay rate** (e.g., 0.5) causes pheromones to evaporate quickly, which encourages exploration of new paths. A **low decay rate** (e.g., 0.99) makes trails more persistent, favoring exploitation of known good paths. A common value is around 0.95.
* `alpha` (Pheromone Influence): This parameter controls the weight given to the pheromone trail when an ant chooses its next step. A **higher `alpha`** makes ants more likely to follow existing strong trails, leading to faster convergence (exploitation). If `alpha` is 0, ants will not consider pheromones at all.
* `beta` (Heuristic Influence): This parameter controls the weight given to the heuristic information, which in this case is the inverse of the distance (favoring shorter moves). A **higher `beta`** makes ants more "greedy," preferring to move to the nearest unvisited city. This focuses the search on locally optimal moves. If `beta` is 0, ants will ignore distance and only follow pheromones.

The balance between **`alpha`** and **`beta`** is crucial. It manages the trade-off between *exploitation* (following known good paths reinforced by pheromones) and *exploration* (trying new, potentially shorter paths).

## Functions

### load_and_filter_data

---

### calculate_distance_matrix

---

### run_aco_with_history

### Ant Engine

This section of code is the decision-making engine for each individual ant in the simulation. Its purpose is to build a complete tour by iteratively selecting the next delivery point to visit. This process continues until every point has been visited exactly once. The selection is not random; it's a probabilistic choice guided by two key factors: the strength of the pheromone trail on a given path and the heuristic value of that path (which is based on its distance).

#### Code Breakdown

Here is a line-by-line explanation of the provided code block:

```python
while len(visited) < n_points:
    current_node = path[-1]
    # Calculate probabilities for the next move:
    pheromone_levels = np.copy(pheromone[current_node])
    pheromone_levels[list(visited)] = 0  # Don't revisit nodes
    heuristic_values = 1.0 / (distances[current_node] + 1e-10)  # Inverse distance; Favor shorter distances; Avoid x/0
    # Probability ∝ (pheromone^alpha) * (heuristic^beta)
    move_probabilities = (pheromone_levels ** alpha) * (heuristic_values ** beta) # Probability of move to next node
    sum_probs = np.sum(move_probabilities)
    if sum_probs == 0:
        # If all probabilities are zero, pick randomly among unvisited
        unvisited = list(set(range(n_points)) - visited)
        next_node = np.random.choice(unvisited)
    else:
        move_probabilities /= sum_probs  # Normalize to sum to 1
        next_node = np.random.choice(range(n_points), p=move_probabilities)
    path.append(next_node)
    visited.add(next_node)
```

1. **`while len(visited) < n_points:`**
    * This loop is the primary mechanism for constructing an ant's path. It continues to run as long as the number of visited points is less than the total number of points, ensuring the ant builds a complete tour.
2. **`current_node = path[-1]`**
    * This identifies the ant's current location by retrieving the last element from the `path` list.
3. **`pheromone_levels = np.copy(pheromone[current_node])`**
    * The ant "looks" at all possible paths leading from its current location. This line retrieves the pheromone levels for all edges connected to the `current_node`. A copy is made so the original pheromone matrix isn't altered.
4. **`pheromone_levels[list(visited)] = 0`**
    * This is a crucial rule in the Traveling Salesman Problem: a city cannot be visited more than once. This line enforces the rule by setting the pheromone level to zero for all nodes that are already in the `visited` set. This effectively makes it impossible for the ant to choose a path to a location it has already been to.
5. **`heuristic_values = 1.0 / (distances[current_node] + 1e-10)`**
    * This calculates the "heuristic desirability" of moving to each node. In ACO for TSP, this is typically the inverse of the distance. Shorter paths have a higher heuristic value, making them more attractive.
    * A very small number (`1e-10`) is added to the distance to prevent division-by-zero errors in the rare case that a distance is 0.
6. **`move_probabilities = (pheromone_levels ** alpha) * (heuristic_values ** beta)`**
    * This is the core formula of the Ant Colony Optimization algorithm. It combines the two main factors for decision-making:
        * `pheromone_levels ** alpha`: The influence of the pheromone trail. The `alpha` parameter controls how much weight is given to the pheromone level.
        * `heuristic_values ** beta`: The influence of the heuristic information (distance). The `beta` parameter controls how much weight is given to the path length.
    * The result, `move_probabilities`, is an array of attractiveness scores for each potential next move.
7. **`if sum_probs == 0:`**
    * This is a conditional check. If the sum of all calculated probabilities is zero (a rare edge case where no valid path can be found), the ant will choose its next node randomly from the pool of unvisited nodes. This prevents the algorithm from getting stuck.
8. **`else:`**
    * This is the standard path.
    * **`move_probabilities /= sum_probs`**: The attractiveness scores are normalized, converting them into a true probability distribution where the sum of all values equals 1.
    * **`next_node = np.random.choice(range(n_points), p=move_probabilities)`**: The ant makes its choice. Instead of just picking the single best option (a greedy approach), it makes a weighted random choice based on the probabilities. Paths with higher scores are more likely to be chosen, but less attractive paths still have a chance. This encourages exploration of new routes.
9. **`path.append(next_node)`**
    * The chosen `next_node` is added to the end of the ant's `path` list.
10. **`visited.add(next_node)`**
    * The `next_node` is also added to the `visited` set, so it won't be considered in future iterations of the loop.

### Total Ant Distance

This code snippet is a key component of the Ant Colony Optimization (ACO). Its purpose is to calculate the total length of a single tour (a potential solution) found by one "ant." In the context of the Traveling Salesman Problem (TSP), a tour is a path that visits every specified delivery point exactly once before returning to the starting point (the depot).

This calculation is performed for every ant in the colony during each iteration of the algorithm. The results are then used to determine the best path found in that iteration and to update the pheromone levels on the travel routes.

#### Code Breakdown

Here is a line-by-line explanation of the provided code block:

```python
# Calculate total distance of the path (including return to start)
current_distance = 0
for j in range(n_points - 1):
    current_distance += distances[path[j], path[j+1]]
current_distance += distances[path[-1], path[0]]  # Complete the tour
all_paths.append((path, current_distance))
```

1. **`current_distance = 0`**
    * This line initializes a variable to store the total distance of the path for the current ant. It starts at zero before any distances are added.
2. **`for j in range(n_points - 1):`**
    * This initiates a loop that iterates through all the segments of the path, except for the final return trip to the depot.
    * `n_points` is the total number of locations to visit (the depot plus all delivery stops from the CSV file).
    * `path` is a list containing the sequence of points the ant has visited, e.g., `[0, 5, 2, 8, ...]`, where each number is the index of a location.
3. **`current_distance += distances[path[j], path[j+1]]`**
    * This is the core of the loop. For each step `j` in the path, it looks up the distance between the current point (`path[j]`) and the next point (`path[j+1]`).
    * The `distances` variable is a 2D matrix that holds the pre-calculated Euclidean distance between every pair of points. This matrix is generated from the latitude and longitude coordinates in the provided dataset.
    * The `+=` operator adds the distance of this segment to the `current_distance` total.
4. **`current_distance += distances[path[-1], path[0]]`**
    * After the loop completes, this line calculates the distance of the final leg of the journey. A valid tour in the TSP requires returning to the start.
    * `path[-1]` gets the last delivery stop in the ant's path.
    * `path[0]` gets the starting point (the depot).
    * This adds the distance from the final stop back to the depot, thus "completing the tour."
5. **`all_paths.append((path, current_distance))`**
    * Finally, this line saves the complete path and its total distance.
    * `all_paths` is a list that stores the results for every ant in the current iteration.
    * A tuple, containing both the `path` list and its calculated `current_distance`, is added to the `all_paths` list for later analysis.

### Pheromone Iteration Update

This piece of code executes the pheromone update phase, which is a fundamental part of the Ant Colony Optimization (ACO) algorithm's learning process. This update occurs at the end of each iteration, after every ant in the colony has constructed a complete tour. The process has two distinct steps: **evaporation**, where old pheromone trails are weakened, and **deposition**, where new pheromones are laid down on the paths the ants have just traveled. This mechanism allows the colony to collectively learn from its experience, reinforcing shorter routes and gradually forgetting less efficient ones.

#### Code Breakdown

Here is a line-by-line explanation of the provided code block:

```python
# Pheromone evaporation: reduce all pheromones by decay factor
pheromone *= decay
# Deposit pheromone: reinforce edges used in each ant's path, inversely proportional to path length
for path, dist in all_paths:
    for j in range(n_points - 1):
        pheromone[path[j], path[j+1]] += 1.0 / dist
    pheromone[path[-1], path[^0]] += 1.0 / dist
```


#### 1. Pheromone Evaporation

* **`pheromone *= decay`**
    * This line handles the evaporation of pheromones across all paths in the network.
    * `pheromone` is a 2D matrix where each element represents the pheromone intensity on the edge between two points.
    * `decay` is a parameter (e.g., 0.95) that determines the rate of evaporation. By multiplying the entire `pheromone` matrix by this value, the pheromone level on every single path is slightly reduced.
    * This step is crucial for **preventing premature convergence**. It ensures that even highly-trafficked paths lose strength over time, which encourages ants to explore alternative routes instead of getting stuck in a potentially suboptimal local solution. It allows the system to "forget" older, possibly less effective, paths.


#### 2. Pheromone Deposition

* **`for path, dist in all_paths:`**
    * This initiates a loop that iterates through every ant's solution from the current iteration. `all_paths` is a list containing tuples, where each tuple holds the `path` (a list of visited nodes) and the total `dist` (distance) of that path.
* **`for j in range(n_points - 1):`**
    * This inner loop goes through each segment of an individual ant's completed tour, from the starting point to the last stop before returning to the depot.
* **`pheromone[path[j], path[j+1]] += 1.0 / dist`**
    * This is the core of the pheromone deposition process. For each segment of the path traveled by an ant, this line adds new pheromone to that edge.
    * The amount of pheromone deposited is `1.0 / dist`, which is **inversely proportional to the total length of the path**.
    * This means that ants that found **shorter (better) paths deposit a larger amount of pheromone**, while ants that traveled longer (worse) paths deposit less. This reinforces good solutions and makes those paths more attractive to ants in the next iteration.
* **`pheromone[path[-1], path] += 1.0 / dist`**
    * This final line ensures the entire tour is reinforced. It adds pheromone to the last leg of the journey: the path from the final delivery stop (`path[-1]`) back to the starting point (`path`). This completes the pheromone update for one ant's entire circular tour.

### Record Best Iteration Solution 

This section of code is executed at the end of each iteration of the Ant Colony Optimization (ACO) algorithm. Its primary role is to track the algorithm's progress by identifying and recording the best solution found so far. It compares the best path from the current iteration with the best path found in all previous iterations and updates the overall best solution if a new, shorter route is discovered.

#### Code Breakdown

Here is a line-by-line explanation of the provided code block:

```python
# Track the best path found so far
current_shortest_path, current_min_distance = min(all_paths, key=lambda x: x[^1])
if current_min_distance < best_path_distance:
    best_path = current_shortest_path
    best_path_distance = current_min_distance

distance_history.append(best_path_distance)  # Save best distance for this iteration
```

1. **`current_shortest_path, current_min_distance = min(all_paths, key=lambda x: x[^1])`**
    * This line finds the best-performing ant from the current iteration.
    * `all_paths` is a list containing the results of every ant from the iteration. Each result is a tuple `(path, distance)`.
    * The `min()` function is used to find the tuple with the smallest value.
    * The `key=lambda x: x[^1]` part specifies that the comparison should be based on the second element of each tuple, which is the total distance (`dist`).
    * The result is that `current_shortest_path` will hold the path list of the best tour from this iteration, and `current_min_distance` will hold its corresponding distance.
2. **`if current_min_distance < best_path_distance:`**
    * This is the core comparison step. It checks if the best path found in the **current iteration** (`current_min_distance`) is better (i.e., shorter) than the best path found in **any previous iteration** (`best_path_distance`).
    * `best_path_distance` acts as the algorithm's memory, storing the shortest distance found across the entire run so far. It is initialized to infinity before the first iteration.
3. **`best_path = current_shortest_path`**
    * This line is executed only if the `if` condition is true. It updates the overall `best_path` variable to store the new, record-breaking route.
4. **`best_path_distance = current_min_distance`**
    * Similarly, this updates the `best_path_distance` to the new shortest distance. These two variables now hold the best solution found by the algorithm up to this point.
5. **`distance_history.append(best_path_distance)`**
    * This line is for logging and visualization. It appends the current overall best distance (`best_path_distance`) to the `distance_history` list.
    * By doing this at every iteration, the code creates a record of how the best solution improves over time. This list is later used to generate a convergence plot, showing how the algorithm progressively finds better solutions.

---

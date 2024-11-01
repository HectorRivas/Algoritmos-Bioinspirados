import numpy as np
import random
import math

def simulated_annealing(distances, temperature, cooling_rate, stopping_temperature):
    # Generar una solución inicial (permuta aleatoria de ciudades)
    current_solution = np.arange(len(distances))
    np.random.shuffle(current_solution)

    # Evaluar la solución inicial
    current_distance = total_distance(current_solution, distances)

    # Guardar la mejor solución encontrada
    best_solution = np.copy(current_solution)
    best_distance = current_distance

    while temperature > stopping_temperature:
        # Generar una nueva solución (intercambiar dos ciudades al azar)
        new_solution = np.copy(current_solution)
        i, j = random.sample(range(len(distances)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        # Evaluar la nueva solución
        new_distance = total_distance(new_solution, distances)

        # Criterio de aceptación
        if new_distance < current_distance or accept_solution(current_distance, new_distance, temperature):
            current_solution = new_solution
            current_distance = new_distance

        # Actualizar la mejor solución encontrada
        if new_distance < best_distance:
            best_solution = new_solution
            best_distance = new_distance

        # Enfriar la temperatura
        temperature *= cooling_rate

    return best_solution, best_distance

def total_distance(solution, distances):
    """Calcula la distancia total de un camino (solución)."""
    distance = 0
    for i in range(len(solution) - 1):
        distance += distances[solution[i], solution[i + 1]]
    distance += distances[solution[-1], solution[0]]  # Regresar al inicio
    return distance

def accept_solution(current_distance, new_distance, temperature):
    """Determina si aceptar una solución peor con probabilidad."""
    if new_distance < current_distance:
        return True
    else:
        # Calcular la probabilidad de aceptar la solución peor
        acceptance_probability = math.exp(-(new_distance - current_distance) / temperature)
        return acceptance_probability > random.random()

# Ejemplo de uso

distances = np.array([[np.inf, 2, 2, 5, 7],
                      [2, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 3],
                      [5, 8, 1, np.inf, 2],
                      [7, 2, 3, 2, np.inf]])

# Parámetros del algoritmo
initial_temperature = 1000  # Temperatura inicial
cooling_rate = 0.995        # Tasa de enfriamiento
stopping_temperature = 1e-8  # Temperatura de parada

best_solution, best_distance = simulated_annealing(distances, initial_temperature, cooling_rate, stopping_temperature)
print(f"Mejor ruta: {best_solution}")
print(f"Distancia más corta: {best_distance}")

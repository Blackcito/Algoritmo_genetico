import numpy as np
from graficar import graficar

# Etiquetas para las direcciones
directions = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'up_right', 5: 'up_left', 6: 'down_right', 7: 'down_left', 8: 'stop'}

# Clase para representar a los individuos
class Individual:
    def __init__(self, genes,agresividad):
        self.genes = genes / np.sum(genes)
        self.position = self.initialize_position()
        self.agresividad = agresividad
        self.selecion = 0
        self.steps = 0
    
    def initialize_position(self):
        while True:
            pos_x = np.random.randint(0, Matriz_X)
            pos_y = np.random.randint(0, 2)
            if matrix_individuos[pos_x][pos_y] is None:
                matrix_individuos[pos_x][pos_y] = self
                return (pos_x, pos_y)

    def move(self):
        direction = directions[np.random.choice(9, p=self.genes)]
        new_pos = movimientos(self.position, direction)
        if verify_step(new_pos, matrix_individuos, self):
            self.position = new_pos

    def reproduce(self, partner): 
        mutation_rate = np.random.uniform(0.10, 0.40)  # Mutation rate de 10 a 15%
        mask = np.random.rand(9) < 0.5  # Máscara aleatoria para seleccionar los genes

        # Genes del primer hijo
        child1_genes = np.where(mask, self.genes, partner.genes)
        child1_aggressiveness = self.agresividad if np.random.rand() < 0.5 else partner.agresividad

        # Genes del segundo hijo (inverso del primer hijo)
        child2_genes = np.where(mask, partner.genes, self.genes)
        child2_aggressiveness = partner.agresividad if np.random.rand() < 0.5 else self.agresividad

        # Mutación de los genes del primer hijo
        if not np.random.uniform(0.10, 1) > mutation_rate:
            mutation_index = np.random.randint(0, 9)  # Índice del gen a mutar (0-8)
            mutation_value = np.random.uniform(0, 0.15)
            child1_genes[mutation_index] += mutation_value

        # Mutación de los genes del segundo hijo
        if not np.random.uniform(0.10, 1) > mutation_rate:
            mutation_index = np.random.randint(0, 9)  # Índice del gen a mutar (0-8)
            mutation_value = np.random.uniform(0, 0.15)
            child2_genes[mutation_index] += mutation_value

        child1_genes /= np.sum(child1_genes)
        child2_genes /= np.sum(child2_genes)

        child1 = Individual(child1_genes,child1_aggressiveness)
        child1.aggressiveness = child1_aggressiveness

        child2 = Individual(child2_genes,child2_aggressiveness)
        child2.aggressiveness = child2_aggressiveness

        return child1, child2


# Función que actualiza la posición según la dirección elegida
def movimientos(pos, direction):
    if direction == 'up':
        return (pos[0], max(0, pos[1] - 1))
    elif direction == 'down':
        return (pos[0], min(Valor_Y, pos[1] + 1))
    elif direction == 'left':
        return (max(0, pos[0] - 1), pos[1])
    elif direction == 'right':
        return (min(Valor_X, pos[0] + 1), pos[1])
    elif direction == 'up_right':
        return (min(Valor_X, pos[0] + 1), max(0, pos[1] - 1))
    elif direction == 'up_left':
        return (max(0, pos[0] - 1), max(0, pos[1] - 1))
    elif direction == 'down_right':
        return (min(Valor_X, pos[0] + 1), min(Valor_Y, pos[1] + 1))
    elif direction == 'down_left':
        return (max(0, pos[0] - 1), min(Valor_Y, pos[1] + 1))
    elif direction == 'stop':
        return pos

# Función de aptitud: premia a aquellos individuos que llegan más cerca del extremo derecho de la matriz
def fitness(pos):
    return pos[0]

def verify_step(pos, matrix_individuos, individuo):
    global asesinatos_generacion_actual; Valor_Y; Valor_X
    global band
    band =0
    if matrix_individuos[pos[0]][pos[1]] is None:
        if pos[1]==Valor_Y:
            band=1
        return True
    existing_individual = matrix_individuos[pos[0]][pos[1]]
    if individuo.agresividad > existing_individual.agresividad:
        matrix_individuos[pos[0]][pos[1]] = individuo
        asesinatos_generacion_actual += 1
        return True  # Se permite el reemplazo
    return False  # No se permite el reemplazo


def assign_probabilities(best_individuals):
    num_individuals = len(best_individuals)
    probabilities = []
    p = 0.5
    for i, individual in enumerate(best_individuals):
        print("++++++++++")
        print(individual.selecion)
        print("=================")
        probability = p * (1 - p) ** i  # Probabilidad proporcional al orden de llegada
        individual.selecion = probability
        print(individual.selecion)


# Inicializaciones matplotlib (listas, figuras, etc.)
average_fitnesses = []
final_positions_over_generations = []
final_reached_counts = []
agresividad_promedio = []
asesinatos_generacion = []

# Solicitudes de datos
num_generations = 2
num_individuals = 50
num_steps = 40
resta_steps = 5
cantidad_generacion_grafica = 1
Matriz_X = 20
Matriz_Y = 20
Valor_X = Matriz_X - 1
Valor_Y = Matriz_Y - 1

#Geneacion semilla

#seed = np.random
np.random.seed()

if num_individuals > Matriz_X*2:
    num_individuals = Matriz_X



# Inicialización de matriz de individuos
matrix_individuos = np.empty((Matriz_X, Matriz_Y), dtype=object)

# Generación inicial de individuos
population = [Individual(genes,np.random.uniform(0.10, 0.50)) for genes in np.random.rand(num_individuals, 9)]



### !!!!! ALGORITMO !!!!!! ####

for generation in range(num_generations):
    final_positions = [] 
    matrix_individuos.fill(None)  # Reiniciar la matriz de individuos en cada generación
    asesinatos_generacion_actual = 0  # Reiniciar el contador de asesinatos para cada individuo
    Pasos_totales=[]
    for individual in population:
        band=0
        for _ in range(num_steps):
            new_pos = individual.move()
            individual.steps += 1
            if(band ==1):
                Pasos_totales.append(individual.steps)
                break
        final_positions.append(individual.position)

        # Actualizar el registro de individuos en la matriz
        matrix_individuos[individual.position[0], individual.position[1]] = individual
        
    print(Pasos_totales)
    print(len(Pasos_totales))
    # Usamos generadores en lugar de listas donde sea posible
    final_positions_over_generations.append(matrix_individuos.copy())  # Guardamos la copia de la matriz
    average_fitnesses.append(np.mean([fitness(pos) for pos in final_positions]))
    final_reached_counts.append(len(np.where(matrix_individuos[Valor_X] != None)[0]))

    best_individuals = [ind for pos, ind in zip(final_positions, population) if pos[0] == Valor_X]
    best_individuals = best_individuals[:Matriz_X]
    probabilities = assign_probabilities(best_individuals)
    # Verificador gente final
    if len(best_individuals) < 2:
        population = [Individual(genes,np.random.uniform(0.10, 0.50)) for genes in np.random.rand(num_individuals, 9)]
        continue

    # Cálculo del promedio de agresividad y asesinatos
    asesinatos_generacion.append(asesinatos_generacion_actual)

    # Asignar probabilidades


    # Reproducción
    new_population =  best_individuals # la nueva generación debe incluir a los padres


    while len(new_population) < num_individuals:
        parent_indices = np.random.choice(len(best_individuals), size=2, replace=False).tolist()
        parent1, parent2 = best_individuals[parent_indices[0]], best_individuals[parent_indices[1]]

        child1, child2 = parent1.reproduce(parent2)
        new_population.append(child1)
        new_population.append(child2)

    # Actualización de la población
    population = np.array(new_population)

    if generation % 50 == 0 and num_steps > 25:
        num_steps -= resta_steps



#### Animacion, Graficacion, Etc #####


graficar(final_positions_over_generations,num_generations,cantidad_generacion_grafica,average_fitnesses,asesinatos_generacion,final_reached_counts)
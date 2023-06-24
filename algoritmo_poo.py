import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Etiquetas para las direcciones
directions = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'up_right', 5: 'up_left', 6: 'down_right', 7: 'down_left'}

# Clase para representar a los individuos
class Individual:
    def __init__(self, genes):
        self.genes = genes / np.sum(genes)
        self.position = (0, 0)

    def move(self):
        direction = directions[np.random.choice(8, p=self.genes)]
        new_pos = move(self.position, direction)
        if verify_step(new_pos, matrix):
            self.position = new_pos

    def reproduce(self, partner):

        self.position = (0,0)
        mask = np.random.rand(8) > 0.5

        # Mutación de los genes
        mutation_rate = np.random.uniform(0.10, 0.15)  # Mutation rate de 10 a 15%
        mutation_mask = np.random.rand(8) < mutation_rate
        mutation_values = np.random.rand(8) * mutation_mask

        child1_genes = np.where(mask, self.genes, partner.genes) + mutation_values
        child2_genes = np.where(mask, partner.genes, self.genes) + mutation_values

        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)
        return child1, child2

# Función que actualiza la posición según la dirección elegida
def move(pos, direction):
    if direction == 'up':
        return (pos[0], max(0, pos[1] - 1))
    elif direction == 'down':
        return (pos[0], min(19, pos[1] + 1))
    elif direction == 'left':
        return (max(0, pos[0] - 1), pos[1])
    elif direction == 'right':
        return (min(Valor_X, pos[0] + 1), pos[1])
    elif direction == 'up_right':
        return (min(Valor_X, pos[0] + 1), max(0, pos[1] - 1))
    elif direction == 'up_left':
        return (max(0, pos[0] - 1), max(0, pos[1] - 1))
    elif direction == 'down_right':
        return (min(Valor_X, pos[0] + 1), min(Valor_X, pos[1] + 1))
    elif direction == 'down_left':
        return (max(0, pos[0] - 1), min(Valor_X, pos[1] + 1))

# Función de aptitud: premia a aquellos individuos que llegan más cerca del extremo derecho de la matriz
def fitness(pos):
    return pos[0]

def verify_step(pos, matrix):
    if pos[0] == Valor_X and matrix[pos[0], pos[1]] > 0:
        return False
    return True


#Inicializaciones matplotlib (listas,figuras,etc)
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
average_fitnesses = []
final_positions_over_generations = []
final_reached_counts = []


#Solicitudes de datos
num_generations = 20
num_individuals = 100
num_steps = 80
Matriz_X = 20
Matriz_Y = 20
Valor_X = Matriz_X - 1
Valor_Y = Matriz_Y - 1

#Inicializacion de matrices a usar, genes, normales, etc, configuracion antes del algoritmo

population = [Individual(genes) for genes in np.random.rand(num_individuals, 8)]

### !!!!! ALGORITMO !!!!!! ####

for generation in range(num_generations):
    final_positions = []
    matrix = np.zeros((Matriz_X, Matriz_Y))  # La matriz se inicializa aquí
    for individual in population:
        for _ in range(num_steps):
            individual.move()
        final_positions.append(individual.position)
        matrix[individual.position[0], individual.position[1]] += 1
    # Usamos generadores en lugar de listas donde sea posible
    final_positions_over_generations.append(matrix.copy())  # Guardamos la copia de la matriz
    average_fitnesses.append(np.mean([fitness(pos) for pos in final_positions]))
    final_reached = [pos for pos in final_positions if pos[0] == Valor_X]
    final_reached_counts.append(len(final_reached))

    best_individuals = [ind for pos, ind in zip(final_positions, population) if pos[0] == Valor_X]

    #verificador gente final

    if len(best_individuals) < 2:
        population = [Individual(genes) for genes in np.random.rand(num_individuals, 8)]
        continue

    #Reproduccion
    new_population = best_individuals.copy()  # la nueva generación debe incluir a los padres
    while len(new_population) < num_individuals:
        parent_indices = np.random.choice(len(best_individuals), size=2, replace=False).tolist()
        parent1, parent2 = best_individuals[parent_indices[0]], best_individuals[parent_indices[1]]
        child1, child2 = parent1.reproduce(parent2)
        new_population.append(child1)
        if len(new_population) < num_individuals:
            new_population.append(child2)

    # Actualización de la población
    population = new_population

#### Animacion, Graficacion, Etc #####

# Configuración de la animación
im = axs[1, 1].imshow(final_positions_over_generations[0], cmap='Reds', interpolation='nearest')
plt.colorbar(im, ax=axs[1, 1])
axs[1, 1].axis('off')

# Función de inicialización para la animación
def init():
    im.set_data(np.rot90(final_positions_over_generations[0]))
    return [im]

# Función de animación que se ejecutará en cada cuadro
def animate(i):
    im.set_data(np.rot90(final_positions_over_generations[i]))
    return [im]

# Crea la animación usando las funciones anteriores
ani = animation.FuncAnimation(fig, animate, frames=num_generations, init_func=init, blit=True)

# Guarda la animación como un archivo .gif
ani.save('evolution.gif', writer='pillow')

# Dibuja la proporción de individuos que alcanzan el final en cada generación
axs[0, 0].plot(average_fitnesses)
axs[0, 0].set_xlabel('Generation')
axs[0, 0].set_ylabel('Average Fitness')

# Nueva gráfica
axs[0, 1].plot(final_reached_counts)
axs[0, 1].set_xlabel('Generation')
axs[0, 1].set_ylabel('Number Reaching Final')

# Dibuja la matriz de posiciones finales para la última generación
axs[1, 0].imshow(np.rot90(final_positions_over_generations[-1]), cmap='Reds', interpolation='nearest')
axs[1, 0].set_title('Final Positions in Last Generation')

# Ajusta el espaciado entre subplots
fig.tight_layout()

# Obtener el administrador de la ventana de la figura
fig_manager = plt.get_current_fig_manager()

# Alternar la visualización en pantalla completa
fig_manager.full_screen_toggle()

# Muestra la figura con todas las gráficas
plt.show()

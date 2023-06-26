import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

# Etiquetas para las direcciones
directions = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'up_right', 5: 'up_left', 6: 'down_right', 7: 'down_left'}

# Clase para representar a los individuos
class Individual:
    def __init__(self, genes):
        self.genes = genes / np.sum(genes)
        self.position = (0, 0)
        self.agresividad = np.random.uniform(0.10, 0.50)
        self.vision = np.random.uniform(1,3)

    def move(self):
        direction = directions[np.random.choice(8, p=self.genes)]
        new_pos = move(self.position, direction)
        if verify_step(new_pos, matrix_individuos, matrix_mov,self.agresividad):
            self.position = new_pos

    def reproduce(self, partner):
        mask = np.random.rand(8) > 0.5

        # Mutación de los genes
        mutation_rate = np.random.uniform(0.10, 0.15)  # Mutation rate de 10 a 15%
        mutation_mask = np.random.rand(8) < mutation_rate
        mutation_values = np.random.rand(8) * mutation_mask

        child1_genes = np.where(mask, self.genes, partner.genes) + mutation_values
        child2_genes = np.where(mask, partner.genes, self.genes) + mutation_values

        child1_genes /= np.sum(child1_genes)
        child2_genes /= np.sum(child2_genes)

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

def verify_step(pos, matrix_individuos,matrix_mov ,agresividad):
    if pos[0] == Valor_X and matrix_mov[pos[0]][pos[1]] > 0:
        if matrix_individuos[pos[0]][pos[1]]:
            existing_individual = matrix_individuos[pos[0]][pos[1]]
            if agresividad > existing_individual.agresividad:
                return True  # Se permite el reemplazo
        return False  # No se permite el reemplazo
    return True

#Inicializaciones matplotlib (listas,figuras,etc)
average_fitnesses = []
final_positions_over_generations = []
final_reached_counts = []


#Solicitudes de datos
num_generations = 20
num_individuals = 100
num_steps = 80
resta_steps = 5
cantidad_generacion_grafica = 1
Matriz_X = 20
Matriz_Y = 20
Valor_X = Matriz_X - 1
Valor_Y = Matriz_Y - 1


#Inicializacion de matrices a usar, genes, normales, etc, configuracion antes del algoritmo

population = [Individual(genes) for genes in np.random.rand(num_individuals, 8)]

# Variables para el gráfico de asesinatos
agresividad_promedio = []
asesinatos_generacion = []

### !!!!! ALGORITMO !!!!!! ####


for generation in range(num_generations):
    final_positions = [] 
    matrix_individuos = [[None] * Matriz_Y for _ in range(Matriz_X)]
    matrix_mov = np.zeros((Matriz_X, Matriz_Y))  # La matriz se inicializa aquí
    
    asesinatos_generacion_actual = 0  # Contador de asesinatos en esta generación
    for individual in population:
        individual.position = (0, 0)
        for _ in range(num_steps):
            individual.move()
        final_positions.append(individual.position)

        # Actualizar el registro de movimiento en matrix_mov
        matrix_mov[individual.position[0], individual.position[1]] += 1

    # Verificar si hay individuos en la misma posición y actualizar matrix_individuos si es necesario
    for individual in population:
        new_pos = individual.position
        individuo = matrix_individuos[new_pos[0]][new_pos[1]]
        if individuo is not None:
            if individuo.position == new_pos:
                matrix_individuos[new_pos[0]][new_pos[1]] = individual
                asesinatos_generacion_actual += 1
        else:
            matrix_individuos[new_pos[0]][new_pos[1]] = individual


    # Usamos generadores en lugar de listas donde sea posible
    final_positions_over_generations.append(matrix_mov.copy())  # Guardamos la copia de la matriz
    average_fitnesses.append(np.mean([fitness(pos) for pos in final_positions]))
    final_reached = [pos for pos in final_positions if pos[0] == Valor_X]
    final_reached_counts.append(len(final_reached))

    best_individuals = [ind for pos, ind in zip(final_positions, population) if pos[0] == Valor_X]

    #verificador gente final
    if len(best_individuals) < 2:
        population = [Individual(genes) for genes in np.random.rand(num_individuals, 8)]
        continue

    # Cálculo del promedio de agresividad y asesinatos
    agresividad_promedio_generacion = np.mean([ind.agresividad for ind in best_individuals])
    agresividad_promedio.append(agresividad_promedio_generacion)
    asesinatos_generacion.append(asesinatos_generacion_actual)
    #Reproduccion
    new_population = best_individuals.copy()  # la nueva generación debe incluir a los padres
    if num_individuals < len(new_population)*2:
        num_individuals = len(new_population)*2

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

fig, axs = plt.subplots(2, 3, figsize=(15, 8))


# Configuración de la animación
im = axs[0, 2].imshow(final_positions_over_generations[0], cmap='Reds', interpolation='nearest')
plt.colorbar(im, ax=axs[0, 2])
axs[0, 2].set_title('Final Positions in Last Generation')
axs[0, 2].axis('off')

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


def update_image_and_title(generation):
    axs[0, 0].imshow(np.rot90(final_positions_over_generations[generation]), cmap='Reds', interpolation='nearest')
    axs[0, 0].set_title(f'Final Positions in Generation {generation}')
    fig.canvas.draw_idle()

# Funciones para manejar los eventos de los botones
def on_previous_button_clicked(event):
    global current_generation
    current_generation = max(current_generation - cantidad_generacion_grafica, 0)
    update_image_and_title(current_generation)

def on_next_button_clicked(event):
    global current_generation
    current_generation = min(current_generation + cantidad_generacion_grafica, num_generations - cantidad_generacion_grafica)
    update_image_and_title(current_generation)

# Variables para la interacción interactiva
current_generation = 0  # Generación actual seleccionada

# Agrega los botones y configura sus posiciones
previous_button_ax = fig.add_axes([0.02, 0.47, 0.1, 0.04])
previous_button = Button(previous_button_ax, 'Previous')
previous_button.on_clicked(on_previous_button_clicked)

next_button_ax = fig.add_axes([0.2, 0.47, 0.1, 0.04])
next_button = Button(next_button_ax, 'Next')
next_button.on_clicked(on_next_button_clicked)





# Dibuja la proporción de individuos que alcanzan el final en cada generación
axs[1, 2].plot(average_fitnesses)
axs[1, 2].set_xlabel('Generation')
axs[1, 2].set_ylabel('Average Fitness')

# Dibuja la proporción de individuos que alcanzan el final en cada generación
axs[1, 0].plot(final_reached_counts)
axs[1, 0].set_xlabel('Generation')
axs[1, 0].set_ylabel('Number Reaching Final')
axs[1, 0].set_title('Number of Individuals Reaching the Final')

# Dibuja el promedio de agresividad vs. asesinatos
axs[1, 1].plot(asesinatos_generacion)
axs[1, 1].set_xlabel('Generation')
axs[1, 1].set_ylabel('Killings per Generation')
axs[1, 1].set_title('Average Aggressiveness vs. Killings')

# Dibuja la matriz de posiciones finales para la última generación
axs[0, 1].imshow(np.rot90(final_positions_over_generations[-1]), cmap='Reds', interpolation='nearest')
axs[0, 1].set_title('Final Positions in Last Generation')
axs[0, 1].set_xlabel('Y')
axs[0, 1].set_ylabel('X')

# Ajusta el espaciado entre subplots
fig.tight_layout()
fig.subplots_adjust(hspace=0.5, wspace=0.3,top=0.9)

# Agrega un título general a todas las subgráficas
fig.suptitle('Genetic Algorithm Evolution')

# Muestra la figura con todas las gráficas
plt.show()

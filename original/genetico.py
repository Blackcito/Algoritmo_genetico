import numpy as np
from graficar import graficar
import sys,os
# Etiquetas para las direcciones
directions = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'up_right', 5: 'up_left', 6: 'down_right', 7: 'down_left', 8: 'stop'}

def limpiar_consola():
    # Limpiar la consola
    if sys.platform.startswith('win'):
        # En Windows
        _ = os.system('cls')
    else:
        # En otros sistemas operativos (por ejemplo, macOS, Linux)
        _ = os.system('clear')

# Clase para representar a los individuos
class Individual:
    def __init__(self, genes,agresividad):
        self.genes = genes / np.sum(genes)
        self.position = self.initialize_position()
        self.agresividad = agresividad
        self.selecion = 0
        self.steps = 0
    
    def initialize_position(self):
        matrix_individuos.fill(None)
        while True:
            pos_x = np.random.randint(0, Matriz_X)
            pos_y = np.random.randint(0, 2)
            if matrix_individuos[pos_x][pos_y] is None:
                matrix_individuos[pos_x][pos_y] = self
                return (pos_x, pos_y)

    def move(self):
        direction = directions[np.random.choice(9, p=self.genes)]
        new_pos = movimientos(self.position, direction)
        if verify_step(new_pos, self):
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

def Posiciones_finales(pos):
    return pos[1]

#Verificacion para pasos y asesinatos
def verify_step(pos, individuo):
    global asesinatos_generacion_actual; matrix_individuos
    if matrix_individuos[pos[0]][pos[1]] is None:
        matrix_individuos[individuo.position[0]][individuo.position[1]] = None
        return True
    
    existing_individual = matrix_individuos[pos[0]][pos[1]]
    if individuo.agresividad > existing_individual.agresividad:
        matrix_individuos[individuo.position[0]][individuo.position[1]] = None
        matrix_individuos[pos[0]][pos[1]] = individuo
        asesinatos_generacion_actual += 1
        return True  # Se permite el reemplazo
    return False  # No se permite el reemplazo
    


def assign_probabilities(best_individuals):
    p = 0.3
    for i, individual in enumerate(best_individuals):
        probability = p * (1 - p) ** i  # Probabilidad proporcional al orden de llegada
        individual.selecion = probability

def main(default_data=None):
    # Datos predeterminados
    if default_data is None:
        default_data = {
            'num_generations': 100,
            'num_individuals': 50,
            'num_steps': 60,
            'resta_steps': 5,
            'resta_generacion': 50,
            'cantidad_generacion_grafica': 10,
            'Matriz_X': 30,
            'Matriz_Y': 30,
            'seed': None
        }

    # Mostrar los datos predeterminados
    limpiar_consola()
    print("Datos predeterminados:")
    for key, value in default_data.items():
        print(f"{key}: {value}")
    
    # Pedir la opción de modificación
    opcion = input("¿Desea modificar los datos predeterminados? (s/n): ")

    # Procesar la opción de modificación
    if opcion.lower() == 's':
        # Mostrar las opciones disponibles para modificar
        limpiar_consola()
        print("\nOpciones disponibles:")
        for i, (key, value) in enumerate(default_data.items(), 1):
            print(f"{i}. {key}: {value}")
        
        # Pedir las opciones a modificar
        opciones_modificar = input("\nSeleccione los números de las opciones que desea modificar (separados por comas): ")
        opciones_modificar = opciones_modificar.split(",")
        
        # Verificar y procesar las opciones seleccionadas
        for opcion_modificar in opciones_modificar:
            opcion_modificar = opcion_modificar.strip()
            if opcion_modificar.isdigit():
                opcion_index = int(opcion_modificar) - 1
                if opcion_index in range(len(default_data)):
                    key = list(default_data.keys())[opcion_index]
                    nuevo_valor = input(f"Ingrese el nuevo valor para {key}: ")
                    try:
                        # Actualizar el dato con el nuevo valor
                        default_data[key] = int(nuevo_valor)
                        print(f"El valor de {key} se ha actualizado a {nuevo_valor}.")
                    except ValueError:
                        print("Error: El valor ingresado no es válido.")
                else:
                    print("Error: Opción no válida.")
            else:
                print("Error: Opción no válida.")
    
        # Mostrar los datos actualizados
        limpiar_consola()
        print("\nDatos actualizados:")
        for key, value in default_data.items():
            print(f"{key}: {value}")
        
        # Verificar si el usuario está seguro de los cambios
        confirmacion = input("\n¿Está seguro de los cambios realizados? (s/n): ")
        if confirmacion.lower() != 's':
            # Volver a ejecutar la función de modificación
            main(default_data)
            return default_data
    
    # Llamar a la función principal con los datos actualizados
    ejecutar_algoritmo(default_data)


def ejecutar_algoritmo(datos):

    #globalicacion
    global num_generations 
    global num_individuals 
    global num_steps
    global resta_steps 
    global resta_generacion 
    global cantidad_generacion_grafica
    global Matriz_X 
    global Matriz_Y 
    global Valor_X
    global Valor_Y
    global matrix_individuos
    global asesinatos_generacion_actual
    # Inicializaciones matplotlib (listas, figuras, etc.)
    average_fitnesses = []
    final_positions_over_generations = []
    final_reached_counts = []
    asesinatos_generacion = []
    sobrevivientes_generacion = []
    # Desempaquetar los datos
    num_generations = datos['num_generations']
    num_individuals = datos['num_individuals']
    num_steps = datos['num_steps']
    resta_steps = datos['resta_steps']
    resta_generacion = datos['resta_generacion']
    cantidad_generacion_grafica = datos['cantidad_generacion_grafica']
    Matriz_X = datos['Matriz_X']
    Matriz_Y = datos['Matriz_Y']
    Valor_X = Matriz_X - 1
    Valor_Y = Matriz_Y - 1
    #Geneacion seilla
    if not datos['seed'] == None:
        np.random.seed(datos['seed'])


    #Verificacion individuos maximos
    if num_individuals > Matriz_X*2:
        num_individuals = Matriz_X*2



    # Inicialización de matriz de individuos
    matrix_individuos = np.empty((Matriz_X, Matriz_Y), dtype=object)

    # Generación inicial de individuos
    population = [Individual(genes,np.random.uniform(0.10, 0.50)) for genes in np.random.rand(num_individuals, 9)]

    ### !!!!! ALGORITMO !!!!!! ####

    for generation in range(num_generations):
        final_positions = [] 
        asesinatos_generacion_actual = 0  # Reiniciar el contador de asesinatos para cada individuo
        Individuos_llegaron=0
        best_individuals = []
        matrix_individuos.fill(None)
        for individual in population:
            individual.steps=0
            for _ in range(num_steps):
                individual.move()
                individual.steps += 1
                if(individual.position[1]==Valor_Y):
                    Individuos_llegaron+=1
                    break
            
            #print(agregue)
            final_positions.append(individual.position)
            # Actualizar el registro de individuos en la matriz
            matrix_individuos[individual.position[0], individual.position[1]] = individual




        # Usamos generadores en lugar de listas donde sea posible
        final_positions_over_generations.append(matrix_individuos.copy())  # Posiciones finales
        sobrevivientes_generacion.append(np.count_nonzero(matrix_individuos))
        average_fitnesses.append(np.mean([Posiciones_finales(pos) for pos in final_positions])) #Promedio posicion generacion
        final_reached_counts.append(Individuos_llegaron) #conteo gente que llego al final

        # Cálculo del promedio de agresividad y asesinatos
        asesinatos_generacion.append(asesinatos_generacion_actual)

        #Selecion individuos finales
        best_individuals = [ind for pos, ind in zip(final_positions, population) if pos[1] == Valor_Y]
        best_individuals = best_individuals[:Matriz_Y]

        #Se ordena con respecto a los pasos dados para llegar al final.
        indices_ordenados = np.argsort([individual.steps for individual in best_individuals])
        best_individuals = np.take(best_individuals, indices_ordenados, axis=0)

        #Asignar probabilidades
        assign_probabilities(best_individuals)
        # Verificador gente final
        if len(best_individuals) < 2 and len(best_individuals) > 0:
            population = []
            best_individuo = best_individuals[0]
            population.append(Individual(best_individuo.genes, best_individuo.agresividad))
            population = [Individual(genes,np.random.uniform(0.10, 0.50)) for genes in np.random.rand(num_individuals -1, 9)]
            continue
        if len(best_individuals)==0:
            population = []
            population = [Individual(genes,np.random.uniform(0.10, 0.50)) for genes in np.random.rand(num_individuals, 9)]
            continue            

        # Reproducción
        new_population = []
        new_population.append(best_individuals[0]) # la nueva generación debe incluir a los padres
        matrix_individuos.fill(None)
        # Obtener las probabilidades de cada objeto
        probabilidades = [individual.selecion for individual in best_individuals]
        # Seleccionar 2 objetos basados en las probabilidades
        probabilidades_normalizadas = [p / sum(probabilidades) for p in probabilidades]

        while True:
            parents = np.random.choice(best_individuals, size=2, replace=False, p=probabilidades_normalizadas)
            child1, child2 = parents[0].reproduce(parents[1])
            if len(new_population) < num_individuals:
                new_population.append(child1)
            if len(new_population) < num_individuals:
                new_population.append(child2)
            else:
                break

        # Actualización de la población
        population = np.array(new_population)
        if generation % resta_generacion == 0 and num_steps > Matriz_Y:
            num_steps -= resta_steps



    #### Animacion, Graficacion, Etc #####

    seed_state = np.random.get_state()
    print("Semilla: "+str(seed_state[1][0]))
    graficar(final_positions_over_generations,num_generations,cantidad_generacion_grafica,average_fitnesses,asesinatos_generacion,final_reached_counts,sobrevivientes_generacion)

main()




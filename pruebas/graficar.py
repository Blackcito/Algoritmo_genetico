import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np

def graficar(final_positions_over_generations,num_generations,cantidad_generacion_grafica,average_fitnesses,asesinatos_generacion,final_reached_counts):



    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Configuración de la animación
    im = axs[0, 2].imshow(np.rot90(final_positions_over_generations[0] != None), cmap='Reds', interpolation='nearest')
    plt.colorbar(im, ax=axs[0, 2])
    axs[0, 2].set_title('Final Positions in Last Generation')
    axs[0, 2].axis('off')

    # Función de inicialización para la animación
    def init():
        im.set_data(np.rot90(final_positions_over_generations[0] != None))
        return [im]

    # Función de animación que se ejecutará en cada cuadro
    def animate(i):
        im.set_data(np.rot90(final_positions_over_generations[i] != None))
        return [im]

    # Crea la animación usando las funciones anteriores
    ani = animation.FuncAnimation(fig, animate, frames=num_generations, init_func=init, blit=True)

    # Guarda la animación como un archivo .gif
    ani.save('evolution.gif', writer='pillow')

    def update_image_and_title(generation):
        axs[0, 0].imshow(np.rot90(final_positions_over_generations[generation] != None), cmap='Reds', interpolation='nearest')
        axs[0, 0].set_title(f'Final Positions in Generation {generation}')
        fig.canvas.draw_idle()

    # Funciones para manejar los eventos de los botones
    def on_previous_button_clicked(event):
        nonlocal current_generation
        current_generation = max(current_generation - cantidad_generacion_grafica, 0)
        update_image_and_title(current_generation)

    def on_next_button_clicked(event):
        nonlocal current_generation
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
    axs[0, 1].imshow(np.rot90(final_positions_over_generations[-1] != None), cmap='Reds', interpolation='nearest')
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
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from ipywidgets import interact, IntSlider

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider




def line_chart_example():
    # Генерація даних
    x = np.linspace(-4, 4, 100) # Generate 100 points from 0 to 10

    y = np.sin(x) # Compute the sine of each x point

    # Створення фігури та осей
    fig, ax = plt.subplots(figsize=(10, 5))

    # Побудова графіка
    ax.plot(x, y, label='Синусоїда', color='blue', linewidth=2, linestyle='-')

    # Додавання заголовку та міток осей
    ax.set_title('Графік синусоїди', fontsize=16)
    ax.set_xlabel('X-вісь', fontsize=14)
    ax.set_ylabel('Y-вісь', fontsize=14)
    ax.legend() # Додавання легенди до графіка
    ax.grid(True) # Додавання сітки до графіка

    # Показ графіка
    plt.show()

def bar_chart_example():
    # Генерація даних
    categories = ["Django", "ReactNative", "Python", "DevOps"]
    values = [85, 90, 95, 80]

    # Створення фігури та осей
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Побудова стовпчикової діаграми
    ax1.bar(categories, values, color=['blue', 'green', 'red', 'orange'])
    ax1.set_title('Стовпчикова діаграма', fontsize=16)
    ax1.set_xlabel('Категорії', fontsize=14)
    ax1.set_ylabel('Значення', fontsize=14)

    # Побудова точкової діаграми
    ax2.scatter(categories, values, color='purple', s=100)
    ax2.set_title('Точкова діаграма', fontsize=16)
    ax2.set_xlabel('Категорії', fontsize=14)
    ax2.set_ylabel('Значення', fontsize=14)

    # Показ графіків
    plt.show()

def pie_chart_example():
    # Генерація даних
    labels = ['Python', 'JavaScript', 'C++', 'Java']
    sizes = [40, 30, 20, 10]
    colors = ['gold', 'lightblue', 'lightgreen', 'lightcoral']

    # Створення фігури
    fig, ax = plt.subplots(figsize=(7, 7))

    # Побудова кругової діаграми
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title('Кругова діаграма розподілу мов програмування', fontsize=16)

    # Додавання кола в центр для створення ефекту "пончика"
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    # Збереження графіка у файл
    # plt.savefig(f'charts/pie_chart_{datetime.now()}.png')

    # Показ графіка
    plt.show()

# def interact_for_jupyter():
#     def plot_sine_wave(frequency):
#         x = np.linspace(0, 10, 100)
#         y = np.sin(frequency * x)
#         plt.figure(figsize=(10, 5))
#         plt.plot(x, y)
#         plt.title(f'Sine Wave with Frequency {frequency}')
#         plt.xlabel('X-axis')
#         plt.ylabel('Y-axis')
#         plt.grid(True)
#         plt.show()
#
#     interact(plot_sine_wave, frequency=IntSlider(min=1, max=10, step=1, value=1))
#
#


if __name__ == "__main__":
    # line_chart_example()
    # bar_chart_example()
    # pie_chart_example()
    interact_for_jupyter()
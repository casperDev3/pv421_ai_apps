import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Створення набору даних (Навчання без вчителя працює з даними без міток)
# Формат: [Фізична сила, Магічна енергія] (оцінки від 1 до 100)
# Уявімо, що це характеристики різних ігрових або комікс-персонажів.
X = np.array([
    [90, 10], [85, 15], [95, 5],   # Сильні фізично, мало магії (Танки/Бійці)
    [10, 90], [15, 85], [5, 95],   # Слабкі фізично, сильні в магії (Маги/Еспери)
    [50, 50], [55, 45], [45, 55],  # Збалансовані персонажі
    [88, 12], [12, 88], [52, 48]   # Додаткові точки для реалістичності
])

#2. Ініціалізація та навчання моделі KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # розділяємо на 3 кластери

#3. Навчання моделі на даних (без міток)
kmeans.fit(X)

#4. Отримання результатів кластеризації
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#5. Візуалізація результатів
plt.figure(figsize=(8, 6))
colors = ["red", "blue", "green"]

for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], color=colors[labels[i]], s=100, edgecolors='black')

# Відображення центрів кластерів (центроїдів)
plt.scatter(centroids[:, 0], centroids[:, 1], color='yellow', marker = "*", s=300, edgecolors='black', label='Centroids')

plt.title("Навчання без вчителя: KMeans Кластеризація")
plt.xlabel("Фізична сила")
plt.ylabel("Магічна енергія")
plt.legend()
plt.grid()
# plt.show()

#6. Тестування на нових даних (наприклад, новий персонаж з характеристиками [70, 30])
new_character = np.array([[70, 30]])
predicted_cluster = kmeans.predict(new_character)
print(f"Новий персонаж з характеристиками [70, 30] належить до кластера: {predicted_cluster[0]}")

if __name__ == "__main__":
    print("Hello World")

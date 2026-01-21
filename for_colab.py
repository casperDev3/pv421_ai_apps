# # Імпортуємо магію
# from sklearn import tree
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# # 1. Дані (Класичний датасет квітів Ірис)
# # X - розміри пелюсток, y - вид квітки
# iris = load_iris()
# X, y = iris.data, iris.target
#
# # Розбиваємо: 80% на підручник (вчимося), 20% на екзамен (перевіряємо)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# # 2. Модель (Дерево рішень - найзрозуміліший алгоритм)
# clf = tree.DecisionTreeClassifier()
#
# # 3. Навчання (Вчимо модель шукати закономірності)
# clf = clf.fit(X_train, y_train)
#
# # 4. Передбачення (Екзамен)
# prediction = clf.predict(X_test)
#
# # 5. Результат
# print(f"Точність нашої моделі: {accuracy_score(y_test, prediction) * 100}%")
#
# # БОНУС: Спробуймо самі придумати квітку
# # [довжина чашолистка, ширина чашолистка, довжина пелюстки, ширина пелюстки]
# my_flower = [[5.1, 3.5, 1.4, 0.2]]
# result = clf.predict(my_flower)
# print(f"Це квітка типу: {iris.target_names[result][0]}")


print("test")
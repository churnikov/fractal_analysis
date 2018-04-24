# Фрактальный анализ

1. [Реализация алгоритма вычисления емкостной размерности](https://nbviewer.jupyter.org/github/kuparez/fractal_analysis/blob/master/capacitarian_dimension.ipynb).
2. [Вычисление спектра размерностей Реньи](https://nbviewer.jupyter.org/github/kuparez/fractal_analysis/blob/master/rényi_spec_dim.ipynb).
3. [Применение метода фрактальной сигнатуры](https://nbviewer.jupyter.org/github/kuparez/fractal_analysis/blob/master/fractal_signature.ipynb).
    1. Вычислить площадь поверхности функции градации серого в зависимости от размера ячейки разбиения. Построить график.
    2. Составить вектор характеристик для изображения и сравнить разные текстуры. Составляющие вектора вычисляются как отношение логарифма площади поверхности к логарифму .
    3. Реализовать алгоритм сегментации, основанный на вычислении площадей ячеек разбиения (или их размерностей) и объединении в подмножество ячеек с близкими значениями характеристик. Диапазон значений показывается цветом.
4. [Вычисление мультифрактального спектра с помощью функции плотности.](https://nbviewer.jupyter.org/github/kuparez/fractal_analysis/blob/master/multifractal%20specter%20with%20density%20function.ipynb)
5. [Вычисление мультифрактального спектра с использованием обобщенной статистической суммы.](https://nbviewer.jupyter.org/github/kuparez/fractal_analysis/blob/master/multifractal_spectre_with_stat_sum.ipynb)

Также все методы из jupyter notebooks перенесены в `fracstuff.py` и этот скрипт можно использовать как модуль (если это кому-то нужно)

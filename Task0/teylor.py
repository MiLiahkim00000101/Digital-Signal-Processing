# Задания 2.2

import matplotlib.pyplot as plt
import math as m
import numpy as np

fig, axs = plt.subplots(2, 2)

# 2. Построить график y=tan(x) на интервале от -4π до 4π.

x_1 = np.linspace(-4 * m.pi, 4 * m.pi, 200)
axs[0][0].plot(x_1, np.tan(x_1), label="tan(x)")

axs[0][0].set_ylim([-6,6])
axs[0][0].set_xlabel('x')
axs[0][0].set_ylabel('y')
axs[0][0].set_title('Задание 1: tan(x)')
axs[0][0].grid(True)
axs[0][0].legend()


# 3. Построить график окружности с радиусом R=7.

x_2 = np.linspace(-7, 7, 800)

axs[0][1].plot(x_2, np.sqrt(7**2 - np.square(x_2)), label="upper", color="green")
axs[0][1].plot(x_2, -np.sqrt(7**2 - np.square(x_2)), label="lower", color="red")


axs[0][1].set_xlim([-10, 10])
axs[0][1].set_ylim([-10,10])
axs[0][1].set_box_aspect(1)
axs[0][1].set_title('Задание 3: circle')
axs[0][1].grid(True)
axs[0][1].legend()

# circle1 = plt.Circle((0, 0), 7, color='r', fill=False)
# axs[0][1].add_patch(circle1)

# 5. Построить график функции: 𝑓(𝑥)=𝑒𝑥𝑝(𝑥). Разложение данной функции в ряд Тейлора записывается следующим образом:

x_teylor = np.linspace(-20, 20, 1000)

axs[1][0].plot(x_teylor, 1 + x_teylor + np.power(x_teylor, 2)/2 + np.power(x_teylor, 3)/6 + np.power(x_teylor, 4)/24 + np.power(x_teylor, 5)/120, label="degree 5")
axs[1][0].plot(x_teylor, 1 + x_teylor + np.power(x_teylor, 2)/2 + np.power(x_teylor, 3)/6 + np.power(x_teylor, 4)/24 + np.power(x_teylor, 5)/120 + np.power(x_teylor, 6)/720, label="degree 6")
axs[1][0].plot(x_teylor, 1 + x_teylor + np.power(x_teylor, 2)/2 + np.power(x_teylor, 3)/6 + np.power(x_teylor, 4)/24 + np.power(x_teylor, 5)/120 + np.power(x_teylor, 6)/720 + np.power(x_teylor, 7)/5040, label="degree 7")
axs[1][0].plot(x_teylor, np.exp(x_teylor), label="orig")
axs[1][0].set_xlim([-20, 20])
axs[1][0].set_ylim([-10,10])
axs[1][0].grid(True)
axs[1][0].legend()

# 6. Реализовать разложение в ряд функции синуса 𝑓(𝑥)=sin (𝑥) используя цикл for. Разложение данной функции в ряд Тейлора записывается следующим образом: Построить графики функции при разложении функции в ряд Тейлора для 1, 3 и 7 степени.

def sin_acc_pows(x, n_parts=3):
    res = 0
    for k in range(n_parts):
        res += (-1) ** k * (x ** (2 * k + 1) / m.factorial(2 * k + 1))
    return res 

axs[1][1].plot(x_teylor, 1 * (x_teylor) / m.factorial(1), label="degree 1")
axs[1][1].plot(x_teylor, 1 * (x_teylor) / m.factorial(1) + (-1) * (x_teylor ** (3)) / m.factorial(3), label="degree 3")
axs[1][1].plot(x_teylor, 1 * (x_teylor) / m.factorial(1) + (-1) * (x_teylor ** (3)) / m.factorial(3) + 1 * (x_teylor ** 7) / m.factorial(7), label="degree 7")
axs[1][1].plot(x_teylor, np.sin(x_teylor), label="orig")

axs[1][1].grid(True)
axs[1][1].set_xlim([-10, 10])
axs[1][1].set_ylim([-7,7])
axs[1][1].legend()



plt.show()
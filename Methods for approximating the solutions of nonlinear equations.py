import numpy as np
import matplotlib.pyplot as plt
from math import e


def metoda_bisectiei(a, b, f, epsilon):
    """
    Implementare a metodei bisectiei.
    """
    assert a < b, 'a trebuie sa fie mai mic strict ca b!'
    x_num = (a + b) / 2  # Prima aproximare
    N = int(np.floor(np.log2((b - a) / epsilon)))  # Criteriul de oprire
        # int pentru ca `range` de mai jos nu accepta float

    # ==============================================================
    # Iteratiile algoritmului
    # ==============================================================
    for _ in range(1, N):
        if f(x_num) == 0:
            break
        elif np.sign(f(a)) * np.sign(f(x_num)) < 0:
            b = x_num
        else:
            a = x_num

        x_num = (a + b) / 2

    return x_num, N

# Exercitiul 1

f = lambda x: x**2 - 3  # Declararea functiei. Am ales functia asta pentru ca una din radacini este sqrt(3).
epsilon = 1e-7  # Eroarea maxima acceptata

x_num, _ = metoda_bisectiei(a=1, b=2, f=f, epsilon=epsilon) # am ales intervalul acesta pentru ca stiu ca acolo trebuie sa se afle solutia mea (sqrt(3))
print('Exercitiul 1')
print('Ecuatia: x^2 - 3 = 0')
print('Intervalul: 1, 2')
print('sqrt(3): {:.7f}'.format(x_num))
print('-' * 72)

# Exercitiul 2

print('Exercitiul 2')
f = lambda x: e ** (x - 2)
g = lambda x: np.cos(e ** (x - 2)) + 1
h = lambda x: e ** (x - 2) - (np.cos(e ** (x - 2)) + 1) # functia este continua si strict crescatoare pe intervalul care ne intereseaza pe noi (1, 3)

# Am ales sugestiv intervalul pentru reprezentarea functiilor si a punctului de intersectie
a = -1  # Capat stanga interval
b = 5  # Capat dreapta interval
x = np.linspace(a, b, 100)  # Discretizare interval [a, b]
y = f(x)  # Valorile functiei f pe discretizarea intervalului
y2 = g(x)  # Valorile functiei g pe discretizarea intervalului
plt.figure(0)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y, c='orange')  # Plotarea functiei f
plt.plot(x, y2, c='blue')  # Plotarea functiei g

x_num, _ = metoda_bisectiei(a=1, b=3, f=h, epsilon=epsilon) # caut solutia pentru functia h(x) = 0
# am ales intervalul acela pentru ca functia f este exponentiala, iar functia g este o functie periodica si are valoarea maxima in 2 ((0<=cos(k)<=1) + 1 => g(x)<=2) astfel stim sigur ca au un singur punct de intersectie
print('Ecuatia: e ^ (x - 2) - (cos(e ^ (x - 2)) + 1) = 0')
print('Solutia numerica: x_num = {:.7f}'.format(x_num), 'y_num = {:.7f}'.format(f(x_num)))
plt.plot(x_num, f(x_num), c='red',  marker='o', markersize=6) # Plotarea punctului de intersectie
plt.legend(["f: e^(x-2)", "g: cos(e^(x-2))+1"])
plt.axvline(0, c='black')
plt.axhline(0, c='black')
plt.show()
print('-' * 72)

# Pentru alegerea celor 3 intervale de la exercitiile 3 si 4 am calculat derivatele functiilor (3*x^2 + 12*x + 3, 3*x^2-2*x-2) si astfel am studiat schimbarile de semn si punctele de extrem local.

# Exercitiul 3

print('Exercitiul 3')

f = lambda x: x**3 + 6*(x**2) + 3*x - 10

a = -5
b = 5
x = np.linspace(a, b, 100)  # Discretizare interval [a, b]
y = f(x)  # Valorile functiei f pe discretizarea intervalului
plt.figure(1)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y, c='blue')  # Plotarea functiei f
plt.legend(["f: x^3 + 6*(x^2) + 3*x - 10"])
plt.axvline(0, c='black')
plt.axhline(0, c='black')
plt.show()

epsilon = 1e-5 # eroarea de aproximare (e aceasi pentru ex 3 si 4)

def pozitie_falsa(x0, x1, f, eps):
    N = 0
    assert x0 < x1
    print('\nIntervalul: ', x0, x1)
    condition = True
    while condition:
        x2 = x0 - (x1 - x0) * f(x0) / (f(x1) - f(x0))

        if f(x0) * f(x2) < 0:
            x1 = x2
        else:
            x0 = x2

        N = N + 1
        condition = abs(f(x2)) > eps

    print('Solutie: ', x2)
    print('Numarul de iteratii: ', N)

# am ales cele trei intervale astfel incat teorema de convergenta sa fie satisfacuta
# pentru fiecare interval f(x)=0 are o solutie unica, iar sirul construit prin metoda pozitiei false converge la acea solutie
# solutia unica apartine intervalului -5, 5
pozitie_falsa(-5, -4, f, epsilon)
pozitie_falsa(-3, -1, f, epsilon)
pozitie_falsa(0.1, 2, f, epsilon)
# De asemenea pentru a vedea ca in intervalul respectiv graficul trece prin 0 putem verifica f(x0)*f(x1)<0 (sau =0 daca unul din capete este solutie)
print('-' * 72)

# Exercitiul 4

print('Exercitiul 4')

f = lambda x: x**3 - x**2 - 2*x
a = -3
b = 3
x = np.linspace(a, b, 100)  # Discretizare interval [a, b]
y = f(x)  # Valorile functiei f pe discretizarea intervalului
plt.figure(2)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y, c='blue')  # Plotarea functiei f
plt.legend(["f: x^3 - x^2 - 2*x"])
plt.axvline(0, c='black')
plt.axhline(0, c='black')
plt.show()

def secanta(a, b, x0, x1, f, eps):
    k = 0
    assert a < b
    assert x0 < x1
    x2 = None
    print('\nIntervalul: ', x0, x1)
    #while abs(x1 - x0) >= abs(x0) * eps:  in cazul asta primeam o eroare la linia 155 f(x1)-f(x0) = 0
    while abs(f(x1)) > eps:
        k = k + 1
        x2 = (x0*f(x1) - x1*f(x0)) / (f(x1)-f(x0))
        if x2 < a or x2 > b:
            print('Introduceti alte valori pentru x0 si x1')
            break
        x0 = x1
        x1 = x2
    print('Solutie: ', x2)
    print('Numarul de iteratii: ', k)

# am ales cele trei intervale astfel incat teorema de convergenta sa fie satisfacuta
# pentru fiecare interval f(x)=0 are o solutie unica.
# De asemenea pentru a vedea ca in intervalul respectiv graficul trece prin 0 putem verifica f(x0)*f(x1)<0
secanta(-3, 3, -2, -0.7, f, epsilon)
secanta(-3, 3, -0.5, 1.5, f, epsilon)
secanta(-3, 3, 1.5, 2.5, f, epsilon)

print('-' * 72)

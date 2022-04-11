# Grafische Darstellung
import numpy as np
import matplotlib.pyplot as plt
# Ganz wichtig, sonst wird der Plot nicht angezeigt
# %matplotlib inline        # braucht man nur beim Notebook
# Identische Funktion
def func_id(x):
    return x
# A =[-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
# A = list(range(-60,60,1))
A = np.arange(-6,6,0.1)
# Eine weltberühmte Aktivierungsfunktion: Die Sigmoide
def func_sigmoid(x):
    # Wichtig: Nicht math.exp, sondern np.exp wegen array
    # Operationen verwenden
    return 1.0 / (1.0 + np.exp(-x))

# Initialisierungen
x = 0.2
y = x
eta = 1.0       # Lernrate von bis 2.0, 1.5, 1.0, 0.01

# Startgewicht
weight = -10.0
# Für den Plot
weights = []
errors = []
w_deltas = []
activations = []
activations_2 = []
derivatives = []
derivatives_2 = []
# Tabelle erzeugen
# Print Überschrift
print("Input x = {:.6f}, Gewünschter Output y = {:.2f}".format(x,y))
print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
      .format('Iter', 'x','w','net i',
              'a','y_hat','y','E',"E'",'w delta'))
# Fixe 121 Schritte
for step in range(2210):
    # Net input berechnen
    net_i = weight * x
    # Aktivierung (identische Funktion)
    # activation = func_id(net_i)
    # Aktivierung (sigmoide Funktion)
    activation = func_sigmoid(net_i)

    # Errechneter Output
    y_hat = activation
    # Quadratischer Fehler: Gewünschter - errechneter Output
    error = 0.5 * (y - y_hat) ** 2
    # Gradient
    # derivative = (-1.0) * x * (y - y_hat)   # Derivative fuer sigmoede neu berechnen
    # sigmoide' = sigmoide * (1 - sigmoide)
    derivative = (-1.0) * activation * (1.0 - activation) * (y - y_hat)
    # Delta für Gewichtsanpassung
    w_delta = (-1) * derivative * eta
    # Daten für den Plot (weight,error)
    weights.append(weight)
    errors.append(error)
    activations.append(activation)
    derivatives.append(derivative)
    w_deltas.append(w_delta)
    # Ausgabe der Änderungen alle 10 Schritte
    if step % 10 == 0:
        print("{}\t{}\t{:6.2f}\t{:5.2f}\t{:5.2f}"
              "\t{:5.2f}\t{:.2f}\t{:.6f}\t{:.6f}\t{:.2f}"
              .format(step, x, weight, net_i, activation, y_hat,
                      y, error, derivative, w_delta))
    # Dafür machen wir das alles: Gewichtsanpassung = Lernen
    weight += w_delta

for B in A:
    activation_2 = func_sigmoid(B)
    y_hat_2 = activation_2
    activations_2.append(activation_2)
    derivative_2 = (-1.0) * activation_2 * (1.0 - activation_2) * (y - y_hat_2)
    derivatives_2.append(derivative_2)
# Plot erzeugen
# Figure und Subplot
fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()

ax1.plot(weights, errors, label="Fehler")
ax1.plot(weights, w_deltas, label="w deltas")
ax2.plot(A, activations_2, label="Activation")
ax2.plot(A, derivatives_2, label="Derivation")
# ax1.plot(weights, activations, label="Fehler")
# ax1.plot(weights, derivatives, label="derivative")
# Titel
ax1.set_title('Gradientenabstieg')
ax2.set_title('Ableitung Activation-sigmoid')
# Legende
legend = ax1.legend(loc='best', fancybox=True, framealpha=0.5)
legend = ax2.legend(loc='best', fancybox=True, framealpha=0.5)
# https://python-course.eu/matplotlib_subplots.php
# ax1.se
# Raster
plt.style.use('seaborn-whitegrid')
plt.xlabel('Gewicht')
plt.ylabel('Fehler')
# xlabel() ändert nur letzten plot, was muss geändert werden?

plt.show()
print(func_sigmoid(1))

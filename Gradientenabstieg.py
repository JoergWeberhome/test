# Grafische Darstellung
import matplotlib.pyplot as plt
# Ganz wichtig, sonst wird der Plot nicht angezeigt
# %matplotlib inline        # braucht man nur beim Notebook
# Identische Funktion
def func_id(x):
    return x

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
# Tabelle erzeugen
# Print Überschrift
print("Input x = {:.6f}, Gewünschter Output y = {:.2f}".format(x,y))
print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
      .format('Iter', 'x','w','net i',
              'a','y_hat','y','E',"E'",'w delta'))
# Fixe 121 Schritte
for step in range(121):
    # Net input berechnen
    net_i = weight * x
    # Aktivierung (identische Funktion)
    activation = func_id(net_i)
    # Errechneter Output
    y_hat = activation
    # Quadratischer Fehler: Gewünschter - errechneter Output
    error = 0.5 * (y - y_hat) ** 2
    # Gradient
    derivative = (-1.0) * x * (y - y_hat)
    # Delta für Gewichtsanpassung
    w_delta = (-1) * derivative * eta
    # Daten für den Plot (weight,error)
    weights.append(weight)
    errors.append(error)
    w_deltas.append(w_delta)
    # Ausgabe der Änderungen alle 10 Schritte
    if step % 10 == 0:
        print("{}\t{}\t{:6.2f}\t{:5.2f}\t{:5.2f}"
              "\t{:5.2f}\t{:.2f}\t{:.6f}\t{:.6f}\t{:.2f}"
              .format(step, x, weight, net_i, activation, y_hat,
                      y, error, derivative, w_delta))
    # Dafür machen wir das alles: Gewichtsanpassung = Lernen
    weight += w_delta

# Plot erzeugen
# Figure und Subplot
fig, ax1 = plt.subplots()
ax1.plot(weights, errors, label="Fehler")
ax1.plot(weights,w_deltas, label="w deltas")
# Titel
ax1.set_title('Gradientenabstieg')
# Legende
legend = ax1.legend(loc='best', fancybox=True, framealpha=0.5)
# Raster
plt.style.use('seaborn-whitegrid')
# Label
plt.xlabel('Gewicht')
plt.show()

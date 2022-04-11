# Grafische Darstellung
import matplotlib.pyplot as plt
# Ganz wichtig, sonst wird der Plot nicht angezeigt
# %matplotlib inline
def entscheidung( summe ):
    """ Berechnung der Entscheidung zum Wert summe
    Input: summe
    Output: 1, falls summe >= 1,
    0 sonst
    """
    if summe >= 1:
        return 1
    else:
        return 0
#------------------------------
# x-Werte des Graphen
x = [-2,-1,0,0.999,1,2,5]
# y-Werte mit der Funktion entscheidung berechnen und mithilfe
# einer List Comprehension eine neue Liste erzeugen (siehe Anhang A)
y = [ entscheidung(i) for i in x ]
# print(y)
# Erzeugen des Graphen mit einer orangefarbigen Stufe und der Bezeichnung step
plt.step(x, y, color='Orange', label='step')

# Die Achsen setzen
plt.grid(True)
# Die horizontale und vertikale 0-Achse etwas dicker in Schwarz zeichnen
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

# Achsenbeschriftung und Titel
plt.xlabel('Summe')
plt.ylabel('Ergebnis')
plt.title('Stufenfunktion')

# Legendenplatzierung festlegen
plt.legend(loc='center right')

# Den Graphen anzeigen
plt.show()

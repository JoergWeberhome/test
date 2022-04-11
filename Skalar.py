import numpy as np

a = np.array(5)
print(a)
print('Der Skalar a hat die Dimension %d' % (a.ndim))

v = np.array([0, 1, -1])
print(v)
print('Der Vektor v hat die Dimension %d' % (v.ndim))

m = np.array([[0, 2, -2],
              [0, 0.5, 2]])
print(m)
print('Die Matrix m hat die Dimension %d' % (m.ndim))

t = np.dot(v, v)
print(t)
print('Die Matrix m hat die Dimension %d' % (t.ndim))

print(np.dot(3, 4))
print(np.dot([2j, 3j], [2j, 3j]))
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
print(np.dot(a, b))


# Jetzt eine Funktion, die eine
# Kurzbeschreibung (Docstring) enthält
def convertFahrenheit2Celsius(temp_in_Fahrenheit):
    """ gibt die Temperatur in Grad Celsius zurück"""
    return (temp_in_Fahrenheit - 32) / 1.8


# Ausgabe Kurzbeschreibung mit der help-Funktion
help(convertFahrenheit2Celsius)

print('neue zeile')

# oder Direktzugriff auf den docstring mit __doc__
print("convertFahrenheit2Celsius:", convertFahrenheit2Celsius.__doc__)

print()


def add_values(x, y, z):
    return x + y + z


print(5, 4, 7)
"""
print(add_values(5, 4, 7))
"""
add_values = lambda x, y, z: x + y + z
print(add_values(6, 7, 8))

myValues = [10, 120, 250, 50, 88, 99, 600]
print(myValues)
# Umwandlung in numpy.array
myArray = np.array(myValues)
print(myArray)
print(myValues[0])
print(myValues[len(myArray) - 1])

A = np.array([[2.1, 0, 6.21, 2.1], [-3.5, 3.45, 9.2, 1.55], [22, 0.45, 3.14, -
32.1]])
print(A)
print()
# ich kann "Spezial"-Arrays erstellen
B = np.ones((3,4)) # erstellt eine 3(Zeilen)x4(Spalten)-Matrix mit 1en
print(B)
print()
# und einige Arrayoperationen
C = A - B # Elementweise Subtraktion
print(C)
print()
# Transponieren einer Matrix, d. h., ich spiegle die Matrix,
# sodass aus einer 3x4-Matrix eine 4x3-Matrix wird
print('Transponieren einer Matrix')
print(A.T)

# die Matrixmultiplikation wird folgendermaßen durchgeführt
Czufall = np.random.random((4, 3))*3  # erstelle eine Matrix mit Zufallszahlen
print()
print('Matrix A')
print(A)
print('Matrix mit Zufallszahlen')
print(Czufall)
C = np.matmul(A, Czufall)
print('Matrixmultiplikation A * C')
print(C)
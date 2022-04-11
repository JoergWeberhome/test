# Eingabewerte (=Ausgabe anderer Neuronen)
# - x1, x2, ... xn
# Gewichtung der Eingabewerte
# - Werte zwischen -1 und 1
# - w1, w2, ... wn

# Ausgabe (Aktivierung)
# - Neuron aktiviert = 1
# - Neuron nicht aktiviert = 0

# Gewichtete Summe der Eingäng berechnet
# Summe(von i=n über n)(wi X xi)
# oftmals BIAS hinzuaddiert
# - Neuronen sollen animiert werden zu feuern
# - w1,w3,...wn
# Summe(von i=n über n)(wi X xi) + bias

# 2. Aktivierungsfunktion
# . groß Phi
# - angewendet auf gewichtete Summe

# Beispiel: Klassifikation
# . Neuron feststellen, ob Tier adoptiert werden soll
# - Bias = 0,2
# - w1=0,1; w2=-0,5; w3=0,1       (von -1 bis 1)
# - Features   x1 = Hund   x2 = kurzhaar   x3 = stubenrein   Output = adoptieren (Label)
#                  0                1               1              0
#                  1                1               1              1
#                  1                0               1              1
#                  1                1               0              0
# adoptieren dann, wenn es ein Huns ist und stubenrein ist
# Gewichtete Summe der Eingänge
# . Summe(von i=n über n)(wi X xi) + bias
# . x1=0; x2=1; x3=1
# . w1=0,1; w2=-0,5; w3=0,1
# . Bias = 0,2
# Einsetzten in die Formel: (0,1 * 0 + (-0,5) * 1 + 0,1 * 1) + 0,2
# = (0 - 0,5 + 0,1) + 0,2
# = -0,4 + 0,2
# = -0,2

# Skalarprodukt
# - Zwei Matrizen: Gewichte und Eingabewerte
# - Komponentenweise Multiplikation
# - Anschließende Addition
# -  beim Skalarprodukt wird jede Zeile multipliziert und dann aufsummiert

# Aktivierungsfunktion
# - bestimmt, wann ein Neuron aktiviert wird (Output=1) und wann nicht (Output=0)
# - Vielzahl von Aktivierungsfunktionen
# : Treppenfunktion (Wenn Schwellwert überschritten, geht Ausgabe sprunghaft nach oben)
#      - Heaviside Function ( bei n < 0 => 0, sonst 1) werden in einfachen neuronalen Netzen verwendet
#   Sigmoidfunktion ( S(n) = 1 / (1 + e^-n)
#      - erzeugt eine S-Kurve
#      - interpretation des Outputs zwischen 0 und 1
#      - Wert >= 0,5 = 1
#      - Wert < 0 = 0
#      - Differenzierbar

# 1. Gewichtete Summe der Eingänge
#  - (w * x) + bias
# 2. Aktivierungsfunktion
#  - phi((w+x)+bias)
#  - Heaviside Function: H(-0,2)=0
#  - Sigmoidfunktion: S(-0,2)=0,450166
#  - Interpredation des Outputs = 0


# hier für ein Trainingsbeispiel [0]
import numpy as np

def heaviside(x):
    if(x<0):
        return 0
    else:
        return 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# input: x1=dog, x2=shorthair, x3=housebroken
training_inputs = np.array([[0,1,1],
                            [1,1,1],
                            [1,0,1],
                            [1,1,0]])       # Matrix [4 , 3]  4 Zeilen und 3 Spalten
training_outputs = np.array([[0,1,1,0]])    # Matrix [1 , 4]  1 Zeilen und 4 Spalten
print('---------')
print('training_inputs = Matrix [4 , 3]  4 Zeilen und 3 Spalten ')
print(training_inputs)
print('training_outputs = Matrix [1 , 4]  1 Zeilen und 4 Spalten')
print(training_outputs)
print('---------')
bias = 0.2
# weights: w1,w2,w3
weights = np.array([[0.2,-0.5,0.2]])        # weigts von 0,1 auf 0,2 geändert und das Ergebnis stimmt!!!!
print('weights = Matrix [1 , 3]  1 Zeilen und 3 Spalten')
print(weights)
print('---------')
# für das Trainingsbeispiel wird jetzt der richtige Output geliefert!!!!!

# wir möchten aufgrund des Inputs den Output berechnen!
# weights = Matrix [1 , 3]  1 Zeilen und 3 Spalten  training_inputs = Matrix [4 , 3]  4 Zeilen und 3 Spalten
weighted_sum = np.dot(weights,training_inputs[0]) + bias
print('Weighted sum:', weighted_sum)

output_heaviside = heaviside(weighted_sum)
print('Output heaviside:',output_heaviside)

output_sigmoid = sigmoid(weighted_sum)
print('Putput sigmoid:', output_sigmoid)
print('len(training_inputs)', len(training_inputs))
for i in range(len(training_inputs)):
    print('---------')
    output_expected = np.array(training_outputs[:, i], ndmin=2).T
    input = np.array(training_inputs[i], ndmin=2).T
    weighted_sum = np.dot(weights, input) + bias
    output_heaviside = heaviside(weighted_sum)
    print('Heaviside Output:')
    print(output_heaviside)

    output_sigmoid = sigmoid(weighted_sum)
    print('Sigmoid Output')
    print(output_sigmoid)

    print('---------')
# der Intervall der Input- und der Outputwerte des Neurons sollte dem Erzeugungsmuster der Aktivierungsfunktion entsprechen!
#  - Heaviside Function: 0 oder 1
#  - Sigmoidfunktion: 0,01 und 0,99
#  - Extremwerte von 0 und 1 können nicht erreicht werden
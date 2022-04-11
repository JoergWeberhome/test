# Pseudocode für einen k-nächste-Nachbarn-Klassifikator
"""
Für alle unbekannten Objekte i mache Folgendes:
Berechne die Distanz von jedem bekannten Objekt zum (unbekannten Objekt) i
Sortiere die Distanzen nach der Größe, beginnend mit der kleinsten Distanz
Bestimme die Häufigkeit der vorkommenden Klassen der ersten k Distanzen
Weise die Klasse mit der höchsten Häufigkeit dem unbekannten Objekt i zu
"""
# die folgenden Module liefern uns …
import numpy as np  # Funktionen zum Sortieren der Distanzen
import numpy.linalg as nl  # Funktionen zur Distanzberechnung
from collections import Counter  # Zählen und Bestimmen der häufigsten Klassen


# unsere kNN-Klassifikator-Klasse
class myKNN:
    """k-Nächste-Nachbarn-Klassifikator"""

    def __init__(self, k_neighbours=3):
        """
        Wird zur Initialisierung des Klassifikators aufgerufen
        """
        self.k_neighbours = k_neighbours
    # Trainingsphase des Klassifikators
    # Der kNN-Klassifikator braucht eigentlich kein Training, da
    # die Klassenzugehörigkeit anhand der nächsten Nachbarn bestimmt wird
    # wir übernehmen nur die Liste Xb der bekannten Objekte in die Klasse


    def fit(self, Xb, y=None):
        """
        Training des Klassifikators mit Liste der bekannten Objekte Xb
        """
        self.Xb = Xb
        self.y = y
        return self


    def predict(self, Xu, y=None):
        """
        Klassifizierung der Liste der unbekannten Objekte Xu
        """
        # zuerst bereiten wir den Resultatsvektor vor
        classindices = []
        for i in Xu:
            # diese Anweisung bestimmt die Distanzen des unbekannten Objekts
            # zu allen bekannten Objekten unserer Liste
            distances = nl.norm(np.transpose(i - self.Xb), axis=0)
            # Sortierung der Distanzen und Liste der ersten k_neighbours-Indizes
            indicesSortedDistances = np.argsort(distances)[:self.k_neighbours]
            # Bestimme die Häufigkeiten der Klassen und gib die häufigste zurück
            mostfrequentClass = \
                Counter(self.y[indicesSortedDistances]).most_common(1)[0][0]
            # Füge das Ergebnis dem Resultatsvektor hinzu
            classindices.append(mostfrequentClass)
        return classindices


# Initialisierung von zwei "Keksen" mit unterschiedlichen Parametern
classificator1 = myKNN(k_neighbours=3)
classificator2 = myKNN(k_neighbours=5)

# Hier nun die x-y-Koordinaten unserer roten und blauen Objekte
# und die dazugehörige Klasse
Xb = np.array([[1, 8.8], [1, 11], [1.2, 15.9], [3.7, 11], [6.1, 8.8],
[9.8, 14.5], [7, 17], [10, 8.1], [11, 10.5], [11.8, 17.5], [16.4, 15.8]])
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# dann das Trainieren
classificator1.fit(Xb,y)
classificator2.fit(Xb,y)
# zum Testen der Klassifikation verwenden wir das grüne Objekt
# mit den Merkmalen (9,9) mit
# unbekannter Klasse und speichern sie unter der Variablen Xu ab
# auch hier verwenden wir das Modul numpy, das wir bei der Definition
# der Klasse bereits importiert haben
Xu = np.array([[9, 9]])
# Und jetzt zur Klassifikation
print("classificator 1: ", classificator1.predict(Xu))
print("classificator 2: ", classificator2.predict(Xu))
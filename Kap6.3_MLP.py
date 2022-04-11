# Für die mathematischen Operationen numpy importieren
import numpy as np
# NEW
from sklearn.utils.validation import check_random_state
# Grafische Darstellung
import matplotlib.pyplot as plt
# Ganz wichtig, sonst wird der Plot nicht angezeigt
# % matplotlib inline


# Die Netzwerkklasse definieren
class MLP(object):

    # Die identische Funktion
    def func_id(self, x):
        return x

    # Eine weltberühmte Aktivierungsfunktion: Die Sigmoide
    def func_sigmoid(self, x):
        # Wichtig: Nicht math.exp sondern np.exp wegen array
        # Operationen verwenden
        return 1.0 / (1.0 + np.exp(-x))

    def __init__(self,
                 n_input_neurons=2,
                 n_hidden_neurons=2,
                 n_output_neurons=1,
                 weights=None,
                 # NEW
                 eta=0.01, n_iterations=10, random_state=2,
                 *args, **kwargs):
        """ Initialisierung des Netzwerkes.
        Wir verwenden eine fixe I-H-O Struktur für den Anfang
             (Input-Hidden-Output)
        Die Anzahl der Neuronen ist flexibel
        Zusätzlich ist es möglich das Netzwerk mit Gewichten
            zu initialisieren[W_IH,W_HO]
        """
        # Anzahl der Neuronen pro Layer
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        # Gewichtsinitialisierung
        self.weights = weights
        W_IH = []
        W_HO = []
        # NEW Lernrate
        self.eta = eta
        # Iterationen
        self.n_iterations = n_iterations
        # NEW Zufallsgenerator
        self.random_state = random_state
        # NEW Erzeugung des Zufallsgenerators (RNG)
        self.random_state_ = check_random_state(self.random_state)
        # NEW Fehler beim fit
        self.errors = []
        # Hier werden alle Daten zur Netzberechnung abgelegt
        self.network = []
        # Input Layer + Bias Neuron: Spalten = n
        #   et_i, a_i, o_i,d_i,delta_i
        self.inputLayer = np.zeros((self.n_input_neurons + 1, 5))
        # Bias Neuron Output ist immer +1
        self.inputLayer[0] = 1.0
        # Den Input Layer zum Netwerk hinzufügen
        self.network.append(self.inputLayer)
        # Weights von Input Layer zum Hidden Layer W_IH
        # Neuron: Zeile x Spalten:
        #   Zeilen = # Hidden, Spalten = # Input
        if weights:
            W_IH = self.weights[0]
        else:
            # NEW
            W_IH = 2 * self.random_state_.random_sample( \
                (self.n_hidden_neurons + 1, self.n_input_neurons + 1)) - 1
        self.network.append(W_IH)
        # NEW Hidden Layer + Bias Neuron:
        # Spalten = net_i,a_i,o_i,d_i,delta_i
        self.hiddenLayer = np.zeros((self.n_hidden_neurons + 1, 5))
        # Bias Neuron Output ist immer +1
        self.hiddenLayer[0] = 1.0
        # Den Hidden Layer zum Netwerk hinzufügen
        self.network.append(self.hiddenLayer)
        # Weights von Hidden Layer zum Output Layer W_HO
        # Neuron: Zeile x Spalten:
        #   Zeilen = # Output, Spalten = # Hidden
        if weights:
            W_HO = self.weights[1]
        else:
            # NEW
            W_HO = 2 * self.random_state_.random_sample( \
                (self.n_output_neurons + 1, self.n_hidden_neurons + 1)) - 1
        self.network.append(W_HO)
        # NEW Output Layer + Bias Neuron:
        # Spalten = net_i,a_i,o_i,d_i,delta_i
        self.outputLayer = np.zeros((self.n_output_neurons + 1, 5))
        # Bias Neuron Output = 0, da nicht relevant.
        # Nur wegen einheitlicher Indizierung vorhanden
        self.outputLayer[0] = 0.0
        # Den Output Layer zum Netwerk hinzufügen
        self.network.append(self.outputLayer)

    def print(self):
        print('Multi-Layer-Perceptron - Netzwerkarchitektur')
        # Insgesamt 7 Stellen, mit drei Nachkommastellen ausgeben
        np.set_printoptions(formatter={'float': lambda x: "{0:7.3f}".format(x)})
        for idx, nn_part in enumerate(self.network):
            print(nn_part)
            print('----------v----------')

    def predict(self, x):
        """ Für Eingabe x wird Ausgabe y berechnet.
        """
        ###############
        # Input Layer
        # Die inputs setzen: Alle Zeilen, Spalte 2
        self.network[0][:, 2] = x
        ###############
        # Hidden Layer
        # Start von Zeile 1 wegen Bias Neuron auf Index Position 0
        # net_j = W_ij . x
        self.network[2][1:, 0] = np.dot(self.network[1][1:, :], \
                                        self.network[0][:, 2])
        # a_j
        self.network[2][1:, 1] = self.func_sigmoid( \
            self.network[2][1:, 0])
        # o_j
        self.network[2][1:, 2] = self.func_id(self.network[2][1:, 1])
        # NEW der_j = o_j*(1-o_j) Ableitung für sigmoide
        self.network[2][1:, 3] = self.network[2][1:, 2] \
                                 * (1.0 - self.network[2][1:, 2])
        ###############
        # Output Layer
        # Start von Zeile 1 wegen Bias Neuron auf 0
        # net_k = = W_jk . h
        self.network[4][1:, 0] = np.dot(self.network[3][1:, :], \
                                        self.network[2][:, 2])
        # a_k
        self.network[4][1:, 1] = self.func_sigmoid( \
            self.network[4][1:, 0])
        # o_k
        self.network[4][1:, 2] = self.func_id(self.network[4][1:, 1])
        # NEW der_k = o_k*(1-o_k) Ableitung für sigmoide
        self.network[4][1:, 3] = self.network[4][1:, 2] \
                                 * (1.0 - self.network[4][1:, 2])
        # Rückgabe Output Vektor
        return self.network[4][:, 2]

    def fit(self, X, Y):
        """ Lernen
        """
        # Gewichtsänderungen
        delta_w_jk = []
        delta_w_ij = []
        # Fehler
        self.errors = []
        # Alle Iterationen
        for iteration in range(self.n_iterations):
            # Für alle Trainingsbeispiele
            error = 0.0
            # for xIdx,x in enumerate(X):
            for x, y in zip(X, Y):
                #####################
                # Vorwärtspfad
                y_hat = self.predict(x)
                # Differenz
                diff = y - y_hat
                # Quadratischer Fehler
                error += 0.5 * np.sum(diff * diff)

                #####################
                # Output Layer
                # delta_k in der Output Schicht = der_k * diff
                self.network[4][:, 4] = self.network[4][:, 3] * diff

                #####################
                # Hidden Layer
                # delta_j in der Hidden Schicht =
                #   der_j * dot(W_kj^T,delta_k)
                self.network[2][:, 4] = \
                    self.network[2][:, 3] * \
                    np.dot(self.network[3][:].T, \
                           self.network[4][:, 4])

                #####################
                # Gewichtsdeltas von W_kj
                # delta_w = eta * delta_k . o_j^T
                delta_w_jk = self.eta * \
                             np.outer(self.network[4][:, 4], \
                                      self.network[2][:, 2].T)
                # Gewichtsdeltas von W_ji
                # delta_w = eta * delta_j . o_i^T
                delta_w_ij = self.eta * \
                             np.outer(self.network[2][:, 4], \
                                      self.network[0][:, 2].T)

                #####################
                # Gewichte anpassen
                self.network[1][:, :] += delta_w_ij
                self.network[3][:, :] += delta_w_jk

            # Sammeln des Fehlers für alle Beispiele
            self.errors.append(error)

    def plot(self):
        """ Ausgabe des Fehlers
        Die im Fehlerarray gespeicherten Fehler als Grafik ausgeben
        """
        # Figure Nummern Start
        fignr = 1
        # Druckgröße in inch
        plt.figure(fignr, figsize=(5, 5))
        # Ausgabe Fehler als Plot
        plt.plot(self.errors)
        # Raster
        plt.style.use('seaborn-whitegrid')
        # Labels
        plt.xlabel('Iteration')
        plt.ylabel('Fehler')


def main():
    # Initialisierung der Trainingsbeispiele
    X = np.array([[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 0]])
    Y = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    # Netzwerk initialisieren
    nn = MLP(eta=0.03, n_iterations=40000, random_state=42)

    # Trainieren des Netzes mit der fit Methode und Ausgabe nach dem Trainieren
    nn.fit(X, Y)
    nn.print()

    # Error Ausgabe als Graph
    nn.plot()

    # Testen der Vorhersage des Trainings Datensatzes
    print('Predict:')
    for x, y in zip(X, Y):
        print('{} {} -> {}'.format(x, y[1], nn.predict(x)[1:2]))


# Hauptprogramm
# Achtung: Benötigt etwas länger, um einen Ausgabe zu produzieren
main()
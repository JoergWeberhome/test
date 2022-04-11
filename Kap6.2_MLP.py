# Für die mathematischen Operationen
import numpy as np
from sklearn.utils.validation import check_random_state


# Die identische Funktion
def func_id(x):
    return x


# Eine weltberühmte Aktivierungsfunktion: Die Sigmoide
def func_sigmoid(x):
    # Wichtig: Nicht math.exp, sondern np.exp wegen array
    # Operationen verwenden
    return 1.0 / (1.0 + np.exp(-x))


# Die rectifier oder ReLU-Aktivierungsfunktion ist an allen Punkten außer bei 0 differenzierbar.
# Bei Werten größer als 0 betrachten wir nur das Maximum der Funktion.
# Dies kann geschrieben werden als: f(x) = max{0, z}
def func_relu(x):
    return np.maximum(x, 0)


# Die Netzwerkklasse definieren
class MLP(object):
    """ Das mehrschichtige Perceptron, MLP """

    def __init__(self,
                 n_input_neurons=2,
                 n_hidden_neurons=2,
                 n_output_neurons=1,
                 weights=None,
                 # NEW
                 eta=0.01, n_iterations=10, random_state=2,
                 *args, **kwargs):
        """ Initialisierung des Netzwerkes
        Wir verwenden eine fixe I-H-O-Struktur für den Anfang:
        (Input-Hidden-Output)
        Die Anzahl der Neuronen ist flexibel
        Zusätzlich ist es möglich, das Netzwerk mit Gewichten zu initialisieren:
        [W_IH,W_HO]
        """
        # Aktivierungs- und Output-Funktion
        self.f_akt = func_sigmoid
        # self.f_akt = func_relu
        self.g_out = func_id
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
        random_state = check_random_state(self.random_state)
        # NEW Fehler beim fit
        self.errors = []
        # Hier werden alle Daten zur Netzberechnung abgelegt
        self.network = []
        # Input-Layer + Bias-Neuron: Spalten = n
        # et_i, a_i, o_i,d_i,delta_i
        self.inputLayer = np.zeros((self.n_input_neurons + 1, 5))
        # Bias-Neuron Output ist immer +1
        self.inputLayer[0] = 1.0
        # Den Input-Layer zum Netzwerk hinzufügen
        self.network.append(self.inputLayer)
        # Weights von Input-Layer zum Hidden Layer W_IH
        # Neuron: Zeile x Spalten: Zeilen = # Hidden, Spalten = # Input
        # Nur initialisieren, falls tatsächlich Gewichte vorhanden
        # Einfache Existenz Prüfung
        if weights:
            W_IH = self.weights[0]
        else:
            # NEW
            W_IH = 2 * random_state.random_sample(
                (self.n_hidden_neurons + 1, self.n_input_neurons + 1)) - 1
        self.network.append(W_IH)
        # NEW Hidden Layer + Bias-Neuron:
        # Spalten = net_i,a_i,o_i,d_i,delta_i

        #  bis hier hin 24.1.2021

        # self.hiddenLayer = np.zeros((self.n_hidden_neurons + 1, 3))
        self.hiddenLayer = np.zeros((self.n_hidden_neurons + 1, 5))
        # Bias-Neuron-Output ist immer +1
        self.hiddenLayer[0] = 1.0
        # Den Hidden Layer zum Netzwerk hinzufügen
        self.network.append(self.hiddenLayer)
        # Weights von Hidden Layer zum Output-Layer W_HO
        # Neuron: Zeile x Spalten: Zeilen = # Output, Spalten = # Hidden
        if weights:
            W_HO = self.weights[1]
        else:
            #  W_HO = np.zeros((self.n_output_neurons + 1, self.n_hidden_neurons + 1))
            # NEW
            W_HO = 2 * random_state.random_sample(
                (self.n_output_neurons + 1, self.n_hidden_neurons + 1)) - 1
        self.network.append(W_HO)
        # NEW Output-Layer + Bias-Neuron:
        # Spalten = net_i,a_i,o_i,d_i,delta_i
        # old             # Output-Layer + Bias-Neuron: Spalten = net_i,a_i,o_i
        # self.outputLayer = np.zeros((self.n_output_neurons + 1, 3))
        self.outputLayer = np.zeros((self.n_output_neurons + 1, 5))
        # Bias-Neuron Output = 0, da nicht relevant
        # Nur wegen einheitlicher Indizierung vorhanden
        self.outputLayer[0] = 0.0
        # Den Output-Layer zum Netzwerk hinzufügen
        self.network.append(self.outputLayer)

    def print(self):
        print('Multi-Layer Perceptron - Netzwerkarchitektur')
        # Insgesamt 7 Stellen, mit drei Nachkommastellen ausgeben
        np.set_printoptions(
            formatter={'float': lambda x: "{0:7.3f}".format(x)})
        for nn_part in self.network:
            print(nn_part)
        print('----------v----------')

    def predict(self, x):
        """ Für Eingabe x wird Ausgabe y_hat berechnet
        Für den Vektor x wird eine Vorhersage berechnet und
        die Matrizenwerte der Layer (nicht Gewichte) werden angepasst
        """
        ###############
        # Input-Layer
        # Die inputs setzen: Alle Zeilen, Spalte 2
        # Input-Layer old
        # old Die Input-Werte setzen: Alle Zeilen, Spalte 0
        # self.network[0][:, 0] = x
        self.network[0][:, 2] = x
        ###############
        # Hidden Layer
        # Start von Zeile 1 wegen Bias-Neuron auf Indexposition 0
        # old net_i
        # net_j = W_ij . x
        # old self.network[2][1:, 0] = np.dot(self.network[1][1:, :],
        #                                 self.network[0][:, 0])
        self.network[2][1:, 0] = np.dot(self.network[1][1:, :],
                                        self.network[0][:, 2])
        # old a_i
        # a_j
        self.network[2][1:, 1] = self.f_akt(
            self.network[2][1:, 0])

        # old o_i
        # o_j
        self.network[2][1:, 2] = self.g_out(self.network[2][1:, 1])
        # NEW der_j = o_j*(1-o_j) Ableitung für sigmoide
        self.network[2][1:, 3] = self.network[2][1:, 2] * \
                                 (1.0 - self.network[2][1:, 2])
        ###############
        # Output-Layer
        # Start von Zeile 1 wegen Bias-Neuron auf 0
        # old net_i
        # net_k = = W_jk . h
        self.network[4][1:, 0] = np.dot(self.network[3][1:, :],
                                        self.network[2][:, 2])
        # old a_i
        # a_k
        self.network[4][1:, 1] = self.f_akt(self.network[4][1:, 0])
        # old o_i
        # o_k
        self.network[4][1:, 2] = self.g_out(self.network[4][1:, 1])
        # NEW der_k = o_k*(1-o_k) Ableitung für sigmoide
        self.network[4][1:, 3] = self.network[4][1:, 2] * \
                                 (1.0 - self.network[4][1:, 2])
        # Rückgabe Output-Vektor
        # old return self.network[4][1:, 2]
        return self.network[4][:, 2]


def main():
    # Initialisierung der Gewichte
    W_IH = np.matrix([[0.0, 0.0, 0.0], [-10, 20.0, 20.0], [30, -20.0, -20.0]])
    W_HO = np.matrix([[0.0, 0.0, 0.0], [-30, 20.0, 20.0]])
    weights = []
    weights.append(W_IH)
    weights.append(W_HO)
    nn = MLP(weights=weights)
    # Netzwerk ausgeben
    nn.print()
    # Test
    X = np.array([[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 0]])
    y = np.array([0, 1.0, 1.0, 0])
    print('Predict:')
    for idx, x in enumerate(X):
        print('{} {} -> {}'.format(x, y[idx], nn.predict(x)))


# Hauptprogramm
main()

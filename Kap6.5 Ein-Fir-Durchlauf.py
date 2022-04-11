# Mathematik
import numpy as np

def main():
    # Trainingsdaten
    X=np.array([[1.0,1.0,1.0]])
    Y=np.array([[0.0,0.0]])
    # Netzwerk initialisieren mit einer Iteration
    nn = MLP(eta=0.03,n_iterations=1,printOn=False,random_state=42)
    # Predict und Netzwerkarchitektur ausgeben
    nn.predict(X[0])
    nn.print()
    # Ein Beispiel lernen und Netzwerkarchitektur ausgeben
    nn.fit(X,Y)
    nn.print()
# Hauptprogramm
main()
import numpy as np
import pandas as pd

# Impostiamo un numero di sessioni (n) per l'esempio
n = 100

# --- Domanda 3: Generazione variabili biometriche ---
battiti = np.random.normal(145, 20, n).clip(60, 200)
sonno = np.random.normal(7.0, 1.2, n).clip(3, 10)
carico = np.random.normal(300, 80, n).clip(50, 600)
riposo = np.random.randint(0, 7, n)

# --- Domanda 4: Calcolo del punteggio di rischio ---
# Sommiamo i pesi convertendo le condizioni booleane in interi
punteggio = (
    2 * (battiti > 170).astype(int) +
    2 * (sonno < 6).astype(int) +
    1 * (carico > 400).astype(int) +
    1 * (riposo == 0).astype(int)
)

# --- Domanda 5: Classificazione del rischio ---
def a_classe(p):
    if p <= 1:
        return 'BASSO'
    elif p <= 3:
        return 'MEDIO'
    else:
        return 'ALTO'

rischio = np.array([a_classe(p) for p in punteggio])

# --- Domanda 6: Creazione DataFrame e Statistiche ---
df = pd.DataFrame({
    'battiti': battiti,
    'sonno': sonno,
    'carico': carico,
    'riposo': riposo,
    'rischio': rischio
})

# Stampa dei risultati
print(f"Numero totale di sessioni: {len(df)}")
print("-" * 30)
print("Distribuzione livelli di rischio:")
for c, n_ in df['rischio'].value_counts().items():
    print(f"{c}: {n_}")

print("-" * 30)
print("Prime 4 righe del DataFrame:")
print(df.head(4))
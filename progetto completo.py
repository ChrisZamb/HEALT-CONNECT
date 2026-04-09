"""
HealthConnect - Sistema di Valutazione del Rischio per Atleti
Progetto RisaVet - Corsi di Intelligenza Artificiale
ITET Rapisardi Da Vinci, Caltanissetta - Classe 3A inf.

Componenti del gruppo:
    Christian Zambuto, Rosario Modica, Emmanuel Zambuto,
    Calogero Cutaia, Carmelo Buscemi, Francesco Tuzzolino
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# ── Generazione dati biometrici simulati ───────────────────────────────────────

n = 100

battiti = np.random.normal(145, 20, n).clip(60, 200)   # frequenza cardiaca (bpm)
sonno   = np.random.normal(7.0, 1.2, n).clip(3, 10)    # ore di sonno
carico  = np.random.normal(300, 80, n).clip(50, 600)   # carico settimanale (min)
riposo  = np.random.randint(0, 7, n)                   # giorni di riposo


# ── Punteggio di rischio ───────────────────────────────────────────────────────

punteggio = (
    2 * (battiti > 170).astype(int) +
    2 * (sonno   <   6).astype(int) +
    1 * (carico  > 400).astype(int) +
    1 * (riposo  ==  0).astype(int)
)


# ── Classificazione ────────────────────────────────────────────────────────────

def a_classe(p: int) -> str:
    """Restituisce la categoria di rischio in base al punteggio."""
    if p <= 1:
        return 'BASSO'
    elif p <= 3:
        return 'MEDIO'
    else:
        return 'ALTO'

rischio = np.array([a_classe(p) for p in punteggio])


# ── DataFrame e statistiche ────────────────────────────────────────────────────

df = pd.DataFrame({
    'battiti': battiti,
    'sonno':   sonno,
    'carico':  carico,
    'riposo':  riposo,
    'rischio': rischio,
})

print(f"Numero totale di sessioni: {len(df)}")
print("-" * 30)
print("Distribuzione livelli di rischio:")
for classe, count in df['rischio'].value_counts().items():
    print(f"  {classe}: {count}")
print("-" * 30)
print("Prime 4 righe del DataFrame:")
print(df.head(4))


# ── Grafico comparativo (valori normalizzati) ──────────────────────────────────

ordine    = ['BASSO', 'MEDIO', 'ALTO']
colori    = ['#2ecc71', '#f39c12', '#e74c3c']
variabili = ['Battiti (bpm)', 'Ore sonno', 'Carico sett. (min)']
massimi   = np.array([200, 10, 600])

medie = df.groupby('rischio')[['battiti', 'sonno', 'carico']].mean().loc[ordine]

fig, ax   = plt.subplots(figsize=(8, 4))
x         = np.arange(len(variabili))
larghezza = 0.25

for i, (classe, colore) in enumerate(zip(ordine, colori)):
    valori      = medie.loc[classe].values
    valori_norm = (valori / massimi) * 100

    ax.bar(x + i * larghezza, valori_norm, larghezza, label=classe, color=colore)

    for j, v in enumerate(valori):
        ax.text(x[j] + i * larghezza, valori_norm[j] + 1,
                f"{v:.1f}", ha='center', fontsize=8)

ax.set_title('Confronto variabili biometriche per livello di rischio')
ax.set_xticks(x + larghezza)
ax.set_xticklabels(variabili)
ax.set_ylabel('Valore normalizzato (% del massimo)')
ax.legend()

plt.tight_layout()
plt.savefig('healthconnect_grafico.png', dpi=150)
plt.show()
print("Grafico salvato: healthconnect_grafico.png")


# ── Preparazione dati per il modello ML ───────────────────────────────────────

le = LabelEncoder()
le.fit(['ALTO', 'BASSO', 'MEDIO'])

y = le.transform(df['rischio'])
X = df[['battiti', 'sonno', 'carico', 'riposo']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nDimensioni Train: {X_train.shape}")
print(f"Dimensioni Test:  {X_test.shape}")


# ── Addestramento e valutazione ────────────────────────────────────────────────

modello = RandomForestClassifier(n_estimators=50, random_state=42)
modello.fit(X_train, y_train)

y_pred      = modello.predict(X_test)
accuratezza = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuratezza:.2%}")


# ── Matrice di confusione ──────────────────────────────────────────────────────

cm           = confusion_matrix(y_test, y_pred)
etichette_cm = le.classes_

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=etichette_cm, yticklabels=etichette_cm)
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.title('Matrice di Confusione - HealthConnect')
plt.tight_layout()
plt.savefig('healthconnect_risultati.png', dpi=150)
plt.show()
print("Grafico salvato: healthconnect_risultati.png")


# ── Report personalizzato atleti ───────────────────────────────────────────────

consigli = {
    'ALTO':  'Riposo obbligatorio.',
    'MEDIO': 'Riduci intensità del 30%.',
    'BASSO': 'Condizione ottimale, allenati!',
}

nomi = ['Marco', 'Sofia', 'Luca']
atleti = pd.DataFrame({
    'battiti': [178, 130, 165],
    'sonno':   [4.5, 8.0,  5.5],
    'carico':  [480, 200,  380],
    'riposo':  [0,   2,    0  ],
}, index=nomi)

pred = le.inverse_transform(modello.predict(atleti))

print("\n--- REPORT ATLETI ---")
for nome, livello in zip(nomi, pred):
    print(f"Atleta: {nome:6s} | Rischio: {livello:5s} | Consiglio: {consigli[livello]}")

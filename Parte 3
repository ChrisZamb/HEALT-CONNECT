import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- (Assumo che tu abbia già df con colonne: battiti, sonno, carico, rischio) ---

# ======================
# DOMANDA 7
# ======================
ordine = ['BASSO', 'MEDIO', 'ALTO']
colori = ['#2ecc71', '#f39c12', '#e74c3c']

medie = df.groupby('rischio')[['battiti', 'sonno', 'carico']].mean().loc[ordine]

variabili = ['Battiti (bpm)', 'Ore sonno', 'Carico sett. (min)']

# ======================
# DOMANDA 8
# ======================
fig, ax = plt.subplots(figsize=(8, 4))

x = np.arange(len(variabili))
larghezza = 0.25

massimi = np.array([200, 10, 600])  # per normalizzazione

for i, (classe, colore) in enumerate(zip(ordine, colori)):
    valori = medie.loc[classe].values
    valori_norm = (valori / massimi) * 100

    barre = ax.bar(x + i * larghezza, valori_norm, larghezza,
                   label=classe, color=colore)

    # Testo sopra le barre (valori reali)
    for j, v in enumerate(valori):
        ax.text(x[j] + i * larghezza, valori_norm[j] + 1,
                f"{v:.1f}", ha='center', fontsize=8)

# ======================
# DOMANDA 9
# ======================
ax.set_title('Confronto variabili biometriche per livello di rischio')
ax.set_xticks(x + larghezza)
ax.set_xticklabels(variabili)

ax.set_ylabel('Valore normalizzato (% del massimo)')
ax.legend()

plt.tight_layout()
plt.savefig('healthconnect_grafico.png')
plt.show()
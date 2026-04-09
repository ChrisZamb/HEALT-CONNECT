# 💪 HEALTHCONNECT — Scrivi il Codice

**Obiettivo:** rispondendo a tutte le domande nell'ordine, avrai scritto l'intero programma funzionante.
Ogni risposta corrisponde a una riga (o blocco) del progetto finale.

---

## PARTE 1 — Import e Setup

**Domanda 1**
Scrivi le cinque righe di import necessarie per caricare: numpy, pandas, matplotlib.pyplot, il classificatore Random Forest da scikit-learn, `train_test_split` e `confusion_matrix` e `accuracy_score` da sklearn, e `LabelEncoder` da `sklearn.preprocessing`.

> 💡 `LabelEncoder` si trova in `sklearn.preprocessing` e serve a convertire stringhe in numeri.

---

**Domanda 2**
Imposta il seme casuale a `7`. Poi definisci `n = 300`.

---

## PARTE 2 — Dataset

**Domanda 3**
Genera le quattro variabili biometriche con distribuzione normale (`.clip()` per limitare i valori):
- `battiti`: media 145, std 20, clip tra 60 e 200
- `sonno`: media 7.0, std 1.2, clip tra 3 e 10
- `carico`: media 300, std 80, clip tra 50 e 600
- `riposo`: interi casuali tra 0 e 6 (inclusi)

> 💡 Usa `np.random.normal(media, std, n).clip(min, max)` per le prime tre.
> Per `riposo` usa `np.random.randint(0, 7, n)`.

---

**Domanda 4**
Calcola il `punteggio` di rischio sommando:
- `2` se battiti > 170
- `2` se sonno < 6
- `1` se carico > 400
- `1` se riposo == 0

Ogni condizione va convertita in intero prima di moltiplicare.

> 💡 Usa `.astype(int)` su ogni condizione booleana, poi moltiplicala per il peso.

---

**Domanda 5**
Definisci la funzione `a_classe(p)` che restituisce:
- `'BASSO'` se `p <= 1`
- `'MEDIO'` se `p <= 3`
- `'ALTO'` altrimenti

Poi crea `rischio = np.array([a_classe(p) for p in punteggio])`.

> 💡 Usa `if / elif / else` — in Python la parola chiave è `elif`, non `else if`.

---

**Domanda 6**
Crea il DataFrame `df` con colonne `battiti`, `sonno`, `carico`, `riposo`, `rischio`.
Poi stampa il numero di sessioni, la distribuzione dei livelli di rischio e le prime 4 righe.

> 💡 Per la distribuzione: `for c, n_ in df['rischio'].value_counts().items(): print(...)`.

---

## PARTE 3 — Grafico EDA

**Domanda 7**
Definisci:
- `ordine = ['BASSO', 'MEDIO', 'ALTO']`
- `colori = ['#2ecc71', '#f39c12', '#e74c3c']`
- Calcola `medie` raggruppando per `rischio` e calcolando la media di `battiti`, `sonno`, `carico` — riordina per `ordine`.
- `variabili = ['Battiti (bpm)', 'Ore sonno', 'Carico sett. (min)']`

> 💡 Usa `df.groupby('rischio')[...].mean().loc[ordine]` per avere le righe nell'ordine corretto.

---

**Domanda 8**
Crea la figura `(8, 4)`, definisci `x = np.arange(len(variabili))` e `larghezza = 0.25`.
Scrivi il ciclo `for` che disegna le barre per BASSO, MEDIO, ALTO.

Per ogni classe:
- normalizza i valori dividendo per `[200, 10, 600]` e moltiplicando per `100`
- disegna le barre con `ax.bar(x + i * larghezza, valori_norm, larghezza, label=classe, color=colore, ...)`
- aggiungi il testo con il valore reale (non normalizzato) sopra ogni barra

> 💡 La normalizzazione serve per confrontare scale diverse (bpm, ore, minuti) sullo stesso grafico.
> Tieni due variabili separate: `valori_norm` per l'altezza delle barre, `valori` per il testo.

---

**Domanda 9**
Completa il grafico: titolo, etichette x (centrate usando `x + larghezza`), etichetta y `'Valore normalizzato (% del massimo)'`, legenda. Salva come `healthconnect_grafico.png` e mostra.

---

## PARTE 4 — Modello ML

**Domanda 10**
Crea il `LabelEncoder`, fittalo su `ordine`, trasforma `df['rischio']` in `y`.
Definisci `X` con le colonne `battiti`, `sonno`, `carico`, `riposo`.
Dividi in train/test con `test_size=0.2` e `random_state=42`.

> 💡 `le.fit(ordine)` impara il mapping. `y = le.transform(df['rischio'])` lo applica.

---

**Domanda 11**
Crea e addestra un `RandomForestClassifier(n_estimators=50, random_state=42)`.
Genera le predizioni e stampa l'accuratezza.

---

## PARTE 5 — Grafico Risultati

**Domanda 12**
Calcola la matrice di confusione. Crea una figura `(5, 4)` e visualizzala con colormap `'YlOrRd'`.
Imposta i tick su `[0, 1, 2]` per entrambi gli assi con le etichette `ordine`.
Aggiungi etichette assi e titolo.

---

**Domanda 13**
Scrivi il ciclo doppio `for i in range(3): for j in range(3):` che scrive il numero al centro di ogni cella.
Il testo è bianco se `cm[i, j] > cm.max() * 0.6`, nero altrimenti. Usa `fontsize=16`.

---

**Domanda 14**
Salva il grafico come `healthconnect_risultati.png` e mostralo.

---

## PARTE 6 — Simulatore

**Domanda 15**
Definisci il dizionario `consigli` con tre chiavi:
- `'ALTO'`: `'Riposo obbligatorio.'`
- `'MEDIO'`: `'Riduci intensita del 30%.'`
- `'BASSO'`: `'Condizione ottimale, allenati!'`

Poi crea il DataFrame `atleti` con 3 atleti (Marco, Sofia, Luca) con questi valori:
- battiti: 178, 130, 165
- sonno: 4.5, 8.0, 5.5
- carico: 480, 200, 380
- riposo: 0, 2, 0

Usa `le.inverse_transform(modello.predict(atleti))` per ottenere le predizioni come stringhe.
Itera con `zip(nomi, pred)` e stampa nome, livello di rischio e consiglio.

> 💡 `le.inverse_transform()` converte i numeri (0,1,2) di ritorno alle stringhe ('BASSO','MEDIO','ALTO').

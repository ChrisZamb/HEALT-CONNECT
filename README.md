# 🏋️ HealthConnect — Valutazione del Rischio per Atleti

> **Progetto RiseVet · Corso di Intelligenza Artificiale · Aragona**

**Componenti del gruppo:**  
Christian Zambuto · Rosario Modica · Emmanuel Zambuto · Calogero Cutaia · Carmelo Buscemi · Francesco Tuzzolino

---

## Descrizione

**HealthConnect** è un sistema di analisi biometrica che valuta il livello di rischio di un atleta prima di una sessione di allenamento.  
A partire da dati fisiologici simulati (frequenza cardiaca, sonno, carico settimanale, giorni di riposo), il sistema:

1. genera e classifica i dati in tre livelli di rischio (**BASSO / MEDIO / ALTO**),
2. visualizza le differenze tra gruppi tramite grafici a barre normalizzati,
3. addestra un modello di **Machine Learning** (Random Forest) per predire il rischio,
4. produce un **report personalizzato** con consigli operativi per ogni atleta.

---

## Struttura del codice

```
healthconnect_risavet.py
│
├── Generazione dati biometrici simulati   (NumPy)
├── Calcolo punteggio di rischio           (pesi booleani)
├── Classificazione BASSO / MEDIO / ALTO
├── DataFrame e statistiche                (Pandas)
├── Grafico comparativo normalizzato       → healthconnect_grafico.png
├── Preparazione dati ML                   (LabelEncoder, train/test split)
├── Addestramento e valutazione            (Random Forest, accuracy)
├── Matrice di confusione                  → healthconnect_risultati.png
└── Report personalizzato atleti
```

---

## Variabili biometriche

| Variabile | Descrizione              | Range simulato |
|-----------|--------------------------|----------------|
| `battiti` | Frequenza cardiaca (bpm) | 60 – 200       |
| `sonno`   | Ore di sonno             | 3 – 10         |
| `carico`  | Carico settimanale (min) | 50 – 600       |
| `riposo`  | Giorni di riposo         | 0 – 6          |

---

## Logica di classificazione del rischio

| Condizione             | Peso |
|------------------------|------|
| Battiti > 170 bpm      | +2   |
| Sonno < 6 ore          | +2   |
| Carico > 400 min/sett. | +1   |
| Nessun giorno di riposo| +1   |

| Punteggio | Livello  |
|-----------|----------|
| 0 – 1     | 🟢 BASSO |
| 2 – 3     | 🟡 MEDIO |
| ≥ 4       | 🔴 ALTO  |

---

## Output generati

| File                          | Contenuto                                        |
|-------------------------------|--------------------------------------------------|
| `healthconnect_grafico.png`   | Grafico a barre normalizzato per livello rischio |
| `healthconnect_risultati.png` | Matrice di confusione del modello ML             |
| Output terminale              | Statistiche, accuratezza, report atleti          |

---

## Requisiti

```
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Installazione dipendenze:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Esecuzione

```bash
python healthconnect_risavet.py
```

Output atteso (esempio):

```
Numero totale di sessioni: 100
------------------------------
Distribuzione livelli di rischio:
  BASSO: 75
  MEDIO: 21
  ALTO: 4
------------------------------
Dimensioni Train: (80, 4)
Dimensioni Test:  (20, 4)
Accuratezza del modello: 100.00%

--- REPORT ATLETI ---
Atleta: Marco  | Rischio: ALTO  | Consiglio: Riposo obbligatorio.
Atleta: Sofia  | Rischio: BASSO | Consiglio: Condizione ottimale, allenati!
Atleta: Luca   | Rischio: MEDIO | Consiglio: Riduci intensità del 30%.
```

---

## Tecnologie utilizzate

| Libreria       | Utilizzo                                       |
|----------------|------------------------------------------------|
| `NumPy`        | Generazione dati simulati e calcoli vettoriali |
| `Pandas`       | Creazione e analisi del DataFrame              |
| `Matplotlib`   | Grafici a barre comparativi                    |
| `Seaborn`      | Heatmap matrice di confusione                  |
| `Scikit-learn` | Preprocessing, Random Forest, metriche ML      |

---


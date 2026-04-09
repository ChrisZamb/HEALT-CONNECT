import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Assicurati che 'df' e 'ordine' siano definiti prima ---
# le = LabelEncoder()
# ordine = ['Basso', 'Medio', 'Alto'] 

# 1. Codifica del target
le = LabelEncoder()
le.fit(ordine)
y = le.transform(df['rischio'])

# 2. Definizione delle feature
X = df[['battiti', 'sonno', 'carico', 'riposo']]

# 3. Split dei dati
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Visualizza i risultati nel terminale di VS Code o nella cella del notebook
print(f"Dimensioni Train: {X_train.shape}")
print(f"Dimensioni Test: {X_test.shape}")




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Creazione del modello con i parametri richiesti
modello = RandomForestClassifier(n_estimators=50, random_state=42)

# 2. Addestramento (fitting) sui dati di train
modello.fit(X_train, y_train)

# 3. Generazione delle predizioni sul set di test
y_pred = modello.predict(X_test)

# 4. Calcolo e stampa dell'accuratezza
accuratezza = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuratezza:.2%}")

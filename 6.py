import pandas as pd

consigli = {
    'ALTO': 'Riposo obbligatorio.',
    'MEDIO': 'Riduci intensita del 30%.',
    'BASSO': 'Condizione ottimale, allenati!'
}


nomi = ['Marco', 'Sofia', 'Luca']
dati = {
    'battiti': [178, 130, 165],
    'sonno': [4.5, 8.0, 5.5],
    'carico': [480, 200, 380],
    'riposo': [0, 2, 0]
}

atleti = pd.DataFrame(dati, index=nomi)

predizioni_numeriche = modello.predict(atleti)
pred = le.inverse_transform(predizioni_numeriche)

print("--- REPORT ATLETI ---")
for nome, livello in zip(nomi, pred):
    consiglio_personalizzato = consigli[livello]
    print(f"Atleta: {nome} | Rischio: {livello} | Consiglio: {consiglio_personalizzato}")
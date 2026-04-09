import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calcolo della matrice di confusione
# Nota: y_true e y_pred devono essere definiti precedentemente nel tuo codice
cm = confusion_matrix(y_true, y_pred)

# Definizione delle etichette
ordine = ['Classe 0', 'Classe 1', 'Classe 2'] # Sostituisci con i nomi reali se necessario

# Creazione della figura (5, 4)
plt.figure(figsize=(5, 4))

# Visualizzazione con heatmap e colormap 'YlOrRd'
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=ordine, yticklabels=ordine)

# Impostazione dei tick su [0, 1, 2] (gestita automaticamente da xticklabels/yticklabels)
# Aggiunta di etichette e titolo
plt.xlabel('Predetto')
plt.ylabel('Reale')
plt.title('Matrice di Confusione - HealthConnect')

# --- Domanda 13 ---
# Salvataggio del grafico
plt.savefig('healthconnect_risultati.png')

# Visualizzazione del grafico
plt.show()
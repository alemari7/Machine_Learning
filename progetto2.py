import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier

file_path = r'C:\Users\39339\Desktop\UNIVERSITA'\MAGISTRALE\ESAMI FATTI\MACHINE LEARNING\PROGETTO\Progetto ML\data.csv'

# Leggi il file CSV e crea il DataFrame
df = pd.read_csv(file_path, low_memory=False)

# Identifica le colonne con dati di tipo object (valori stringa)
object_cols = df.select_dtypes(include=['object']).columns

# Inizializza il LabelEncoder
label_encoder = LabelEncoder()

# Codifica le colonne con valori stringa
df[object_cols] = df[object_cols].apply(lambda col: label_encoder.fit_transform(col))

df.dropna(inplace=True)

columns_to_drop = ['is_cup', 'home_team_history_is_cup_1','home_team_history_is_cup_1','home_team_history_is_cup_2',
                    'home_team_history_is_cup_3','home_team_history_is_cup_4','home_team_history_is_cup_5', 'home_team_history_is_cup_6',
                    'home_team_history_is_cup_7','home_team_history_is_cup_8', 'home_team_history_is_cup_9','home_team_history_is_cup_10',
                    'away_team_history_is_cup_1','away_team_history_is_cup_1','away_team_history_is_cup_2',
                    'away_team_history_is_cup_3','away_team_history_is_cup_4','away_team_history_is_cup_5', 'away_team_history_is_cup_6',
                    'away_team_history_is_cup_7','away_team_history_is_cup_8', 'away_team_history_is_cup_9','away_team_history_is_cup_10']

df = df.drop(columns_to_drop, axis=1)

X_train = df.drop('target', axis=1)
y_train = df['target']

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

""" sfs = SFS(RandomForestClassifier(n_estimators=1, n_jobs=8, random_state=13),
          k_features=10,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=2) """

# sfs.fit(X_train, y_train)

# Salva l'oggetto SFS in un file
# joblib.dump(sfs, 'sfs_model.pkl')

# Carica l'oggetto SFS da un file
sfs_loaded = joblib.load('sfs_model.pkl')

# Utilizza l'oggetto SFS addestrato per ottenere la combinazione di feature selezionata
selected_features = sfs_loaded.k_feature_names_
print(selected_features)

X_train_reduced = sfs_loaded.transform(X_train)
X_test_reduced = sfs_loaded.transform(X_test)


#################################################################  RANDOM FOREST


# Definisci il modello Random Forest con i parametri desiderati
#random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Addestra il modello sul set di addestramento
#random_forest.fit(X_train_reduced, y_train)

# Salva il modello su disco
#joblib.dump(model, 'random_forest_model.pkl')

# Per caricare il modello successivamente:
random_forest = joblib.load('random_forest_model.pkl')

# Esegui la predizione sul set di test
y_pred_RF = random_forest.predict(X_test_reduced)

# Valuta l'accuracy
accuracy_RF = accuracy_score(y_test, y_pred_RF)
print("Accuracy Random Forest:", accuracy_RF)


#################################################################  BAYES


# Crea il classificatore Naive Bayes
# naive_bayes = GaussianNB()

# Addestra il classificatore sul training set
# naive_bayes.fit(X_train_reduced, y_train)

# Salva il modello su disco
# joblib.dump(naive_bayes, 'naive_bayes_model.pkl')

# Per caricare il modello successivamente:
naive_bayes = joblib.load('naive_bayes_model.pkl')

# Effettua la predizione sul test set
y_pred_BY = naive_bayes.predict(X_test_reduced)

# Calcola l'accuracy della predizione
accuracy_BY = accuracy_score(y_test, y_pred_BY)
print("Accuracy Bayes:", accuracy_BY)


#################################################################  SVM

# svm_classifier = SVC(kernel='linear', C=1.0)

# Addestra il classificatore sul training set
# svm_classifier.fit(X_train_reduced, y_train)

# Salva il modello su disco
# joblib.dump(svm_classifier, 'svm_model.pkl')

# Per caricare il modello successivamente:
svm_classifier = joblib.load('svm_model.pkl')

# Effettua la predizione sul test set
y_pred_SVM = svm_classifier.predict(X_test_reduced)

# Calcola l'accuracy della predizione
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
print(f"Accuracy SVM:", accuracy_SVM)


#################################################################  GRAFICO ACCURACY DEI 3 MODELLI


# Calcola le metriche di Precision e Recall per ciascun modello
precision_RF = precision_score(y_test, y_pred_RF, average='weighted')
precision_BY = precision_score(y_test, y_pred_BY, average='weighted')
precision_SVM = precision_score(y_test, y_pred_SVM, average='weighted')

recall_RF = recall_score(y_test, y_pred_RF, average='weighted')
recall_BY = recall_score(y_test, y_pred_BY, average='weighted')
recall_SVM = recall_score(y_test, y_pred_SVM, average='weighted')

# Stampa accuracy, precision e recall dei modelli
print("Random Forest:")
print("Accuracy:", accuracy_RF)
print("Precision:", precision_RF)
print("Recall:", recall_RF)
print()

print("Naive Bayes:")
print("Accuracy:", accuracy_BY)
print("Precision:", precision_BY)
print("Recall:", recall_BY)
print()

print("SVM:")
print("Accuracy:", accuracy_SVM)
print("Precision:", precision_SVM)
print("Recall:", recall_SVM)

# ... (codice per il grafico precedente)

# Accuracy Precision e Recall dei tre modelli
precision_scores = [precision_RF, precision_BY, precision_SVM]
recall_scores = [recall_RF, recall_BY, recall_SVM]
accuracy_scores = [accuracy_RF, accuracy_BY, accuracy_SVM]

# Etichette dei modelli
models = ['Random Forest', 'Naive Bayes', 'SVM']

# Crea il grafico a barre
plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(models))

plt.bar(index, accuracy_scores, bar_width, label='Accuracy', color='blue')
plt.bar(index + bar_width, precision_scores, bar_width, label='Precision', color='green')
plt.bar(index + 2 * bar_width, recall_scores, bar_width, label='Recall', color='orange')

plt.xlabel('Modello')
plt.ylabel('Valore')
plt.title('Confronto Accuracy, Precision e Recall dei Modelli')
plt.xticks(index + bar_width, models)
plt.legend()
plt.ylim(0, 1.0)
plt.show()
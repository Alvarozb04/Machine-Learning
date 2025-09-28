# Librerías para análisis de datos y operaciones matemáticas
import pandas as pd
import numpy as np

# Librerías para gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset (ejemplo con CSV, cámbialo por tu archivo)
df = pd.read_csv("C:/Users/ÁlvaroZB/Desktop/Porros, Farlopa y una Pistola/Materiales/materials_project_dataset.csv")
'''
# VER COLUMNAS
# Mostrar las columnas
print("Columnas del dataset:")
print(df.columns.tolist())

# Mostrar las primeras filas
print("\nPrimeras filas del dataset:")
print(df.head())

# Ejemplo de operación matemática: media de cada columna numérica
print("\nMedia de las columnas numéricas:")
print(df.mean(numeric_only=True))

# Ejemplo de gráfico: histograma de la primera columna numérica
columna = df.select_dtypes(include=np.number).columns[0]  #_
'''

''' 
# Variables que quieres graficar
x_var = "density"  # reemplaza con el nombre de tu columna X
y_var = "energy_above_hull"  # reemplaza con el nombre de tu columna Y

# --- Opción 1: Usando matplotlib ---
plt.scatter(df[x_var], df[y_var], color='blue', alpha=0.6)
plt.title(f"Gráfico de puntos: {x_var} vs {y_var}")
plt.xlabel(x_var)
plt.ylabel(y_var)
plt.show()

# --- Opción 2: Usando seaborn ---
sns.scatterplot(data=df, x=x_var, y=y_var, hue=None)  # hue=None evita colorear por otra variable
plt.title(f"Gráfico de puntos: {x_var} vs {y_var}")
plt.show()
'''
''' #HISTOGRAMA PARA BOOLEANAS
# Nombre de la columna booleana
col = "is_stable"

# Conteo de valores True/False
conteo = df[col].value_counts()

# Crear gráfico de barras
ax = sns.countplot(data=df, x=col, palette="Set2")
plt.title(f"Distribución de {col}")
plt.xlabel("Valor")
plt.ylabel("Número de materiales")

# Añadir etiquetas encima de cada barra
for p in ax.patches:
    altura = p.get_height()
    ax.annotate(f"{altura}", 
                (p.get_x() + p.get_width() / 2., altura), 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.show()
'''
#COMPARAR DOS VARIABLES BOOLENAAS
# Nombres de las columnas booleanas (cámbialos por los que te interesen)
col1 = "is_semiconductor"
col2 = "is_stable"

# Crear tabla de contingencia
resumen = pd.crosstab(df[col1], df[col2])

# Mostrar tabla como heatmap
plt.figure(figsize=(6,4))
sns.heatmap(resumen, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Coincidencias entre {col1} y {col2}")
plt.xlabel(col2)
plt.ylabel(col1)
plt.show()

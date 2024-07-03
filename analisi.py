import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from scipy.cluster.hierarchy import fcluster

# Cargar los datos
file_path = "C:/Users/ben19/Downloads/Articulo/cigarrillos.xlsx"
df = pd.read_excel(file_path)

# Convertir las columnas a tipo numérico si es necesario
df['ipc'] = pd.to_numeric(df['ipc'], errors='coerce')
df['poblacion'] = pd.to_numeric(df['poblacion'], errors='coerce')
df['paquetes'] = pd.to_numeric(df['paquetes'], errors='coerce')
df['ingreso'] = pd.to_numeric(df['ingreso'], errors='coerce')
df['impuesto'] = pd.to_numeric(df['impuesto'], errors='coerce')
df['precio'] = pd.to_numeric(df['precio'], errors='coerce')

# Agrupar por departamento y calcular consumo total de cigarrillos
consumo_por_depto = df.groupby('departamento')[['paquetes', 'precio', 'ingreso']].sum()
print(consumo_por_depto)

# Seleccionar las variables relevantes para el clustering
datos_cluster = consumo_por_depto[['paquetes', 'precio', 'ingreso']]

# Realizar el clustering jerárquico
enlace = linkage(datos_cluster, method='ward')  # Realizamos el enlace utilizando el método de Ward

# Graficar el dendrograma
plt.figure(figsize=(12, 6))  # Definimos el tamaño del gráfico
dendrogram(enlace, labels=consumo_por_depto.index.values, leaf_rotation=90, leaf_font_size=8)  # Graficamos el dendrograma con los nombres de los departamentos
plt.title('Dendrograma de Clustering Jerárquico')  # Título del gráfico
plt.xlabel('Departamentos')  # Etiqueta del eje x
plt.ylabel('Distancia')  # Etiqueta del eje y
plt.show()  # Mostramos el gráfico

# Obtener los clusters
k = 2  # Número de clusters
clusters = fcluster(enlace, k, criterion='maxclust')

# Añadir la columna de clusters al DataFrame
consumo_por_depto['cluster'] = clusters

# Visualizar los resultados del clustering
pca = PCA(n_components=2)  # Inicializamos PCA para reducir la dimensionalidad de los datos a 2 dimensiones
principalComponents = pca.fit_transform(datos_cluster)  # Aplicamos PCA a los datos
principalDf = pd.DataFrame(data=principalComponents, columns=['componente_principal_1', 'componente_principal_2'])  # Creamos un DataFrame con las componentes principales

# Graficar los componentes principales con colores según los clusters jerárquicos
plt.figure(figsize=(10, 6))  # Definimos el tamaño del gráfico
plt.scatter(principalDf['componente_principal_1'], principalDf['componente_principal_2'], c=consumo_por_depto['cluster'], cmap='viridis', s=50)  # Graficamos los puntos con colores según los clusters
for i, txt in enumerate(consumo_por_depto.index):  # Iteramos sobre cada departamento
    plt.annotate(txt, (principalDf['componente_principal_1'][i], principalDf['componente_principal_2'][i]), fontsize=8)  # Anotamos el nombre de cada departamento en el gráfico
plt.title('Clustering Jerárquico de Departamentos según Consumo de Cigarrillos')  # Título del gráfico
plt.xlabel('Componente Principal 1')  # Etiqueta del eje x
plt.ylabel('Componente Principal 2')  # Etiqueta del eje y
plt.grid(True)  # Activar la cuadrícula en el gráfico
plt.colorbar(label='Cluster')  # Añadir la barra de color para identificar los clusters
plt.show()  # Mostrar el gráfico

# Agrupar por año y calcular el consumo total de cigarrillos
consumo_por_anio = df.groupby('año')['paquetes'].sum()

# Convertir el índice a un índice de fechas
consumo_por_anio.index = pd.date_range(start=f'{consumo_por_anio.index[0]}', periods=len(consumo_por_anio), freq='YE')

# Graficar el consumo de cigarrillos por año
plt.figure(figsize=(10, 6))
plt.plot(consumo_por_anio.index, consumo_por_anio.values, marker='o')
plt.title('Consumo Total de Cigarrillos por Año')
plt.xlabel('Año')
plt.ylabel('Consumo de Paquetes')
plt.grid(True)
plt.show()

# Predicción del consumo futuro utilizando un modelo ARIMA
# Ajustar el modelo ARIMA
modelo_arima = ARIMA(consumo_por_anio, order=(1, 1, 1))
modelo_fit = modelo_arima.fit()

# Realizar predicciones
predicciones = modelo_fit.forecast(steps=5)

# Graficar las predicciones
plt.figure(figsize=(10, 6))
plt.plot(consumo_por_anio.index, consumo_por_anio.values, marker='o', label='Datos Observados')
plt.plot(pd.date_range(start=consumo_por_anio.index[-1] + pd.DateOffset(years=1), periods=5, freq='YE'), predicciones, marker='o', linestyle='dashed', label='Predicciones')
plt.title('Predicción del Consumo de Cigarrillos')
plt.xlabel('Año')
plt.ylabel('Consumo de Paquetes')
plt.legend()
plt.grid(True)
plt.show()

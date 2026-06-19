# Proyecto-1-IA-Los-Simpsons
Desarrollo de Proyecto 1 de Introducción a la Inteligencia Artificial 

## Definición del problema

### Claridad y Relevancia

El proyecto está basado en la serie norteamericana "Los Simpsons" que a lo largo de sus 39 años ha experimentado éxitos y grandes valores de sintonía como en la valoración de su audiencia. El problema consiste en analizar qué características de los episodios están asociadas a una mejor recepción por parte del público y desarrollar un modelo capaz de predecir la calificación IMDb de un episodio a partir de variables relacionadas con su producción y desempeño. Entre estas variables se consideran la temporada, el número de episodio, la audiencia en millones de espectadores, los responsables de dirección y escritura. Permitiendo todo esto comprender qué factores influyen directamente en el éxito prolongado del show.

## Plan de acción

### Descripción del dataset y fuente

El dataset contiene información de episodios de Los Simpsons. Cada registro representa un episodio e incluye variables descriptivas relacionadas con su producción, emisión y recepción por parte del público. Las principales columnas disponibles son:

- season: temporada a la que pertenece el episodio.
- number_in_season: número del episodio dentro de la temporada.
- number_in_series: número total dentro de la serie.
- directed_by: director(es) del episodio.
- written_by: guionista(s) responsables.
- us_viewers_in_millions: audiencia en millones de espectadores.
- tmdb_rating: valoración promedio en TMDB.
- tmdb_vote_count: cantidad de votos registrados en TMDB.
- imdb_rating: valoración promedio en IMDb.

Para poder explorar de manera más específica los datos se realizarán análisis exploratorios y correlaciones. Por otro lado, el dataset fue seleccionado y descargado de la plataforma [Kaggle] (https://www.kaggle.com/datasets/jonbown/simpsons-episodes-2016?select=simpsons_episodes.csv). 

## Modelo(s) seleccionado(s) y estrategia de evaluación claramente explicados

Se seleccionó un modelo de Regresión Lineal, utilizando como variable objetivo la calificación IMDb (imdb_rating). El modelo buscará estimar la puntuación de un episodio a partir de las características disponibles en el conjunto de datos, analizando la relación entre las variables predictoras y la calificación de IMDb, así como la capacidad del modelo para estimar dicha puntuación.
Para la evaluación, el conjunto de datos se dividirá en entrenamiento (80%) y prueba (20%). Posteriormente, sobre el conjunto de entrenamiento se aplicará validación cruzada K-Fold con k=5, con el fin de obtener una estimación más robusta del desempeño del modelo antes de realizar la evaluación final sobre el conjunto de prueba. El rendimiento se analizará mediante las métricas R², MAE y RMSE.

### Justifiación de modelo: ventajas, limitaciones y pertinencia.

Se seleccionó la Regresión Lineal debido a que la variable objetivo del estudio, imdb_rating, corresponde a una variable numérica continua. Este tipo de modelo permite analizar y cuantificar la relación existente entre una variable dependiente y múltiples variables explicativas, por lo que resulta adecuado para estimar la valoración de un episodio a partir de sus características. 

Entre sus principales ventajas se encuentran su simplicidad de implementación y facilidad de interpretación. Además, permite identificar qué variables tienen una mayor influencia sobre la calificación IMDb mediante el análisis de los coeficientes asociados a cada predictor. Como limitación, la regresión lineal supone que existe una relación lineal entre las variables analizadas, por lo que puede que no llegue a representar correctamente patrones más complejos. Sin embargo, este modelo resulta pertinente para el problema planteado, ya que permite predecir una variable numérica como la calificación IMDb e identificar qué factores influyen en la valoración de los episodios. Además, destaca por ser un modelo simple, fácil de interpretar y adecuado para un análisis exploratorio inicial.

## Metodología aplicada (paso a paso)

### Paso 1: Carga de librerías y data:

El proyecto parte con la preparación del entorno en Python, cargandao Pandas y NumPy para ordenar las tablas y hacer cálculos, y Matplotlib junto con Seaborn para generar los gráficos. Para la parte del entrenamiento y predicción se incluyen los elementos de la librería Scikit-Learn, además dentro de esta se incluye la validación cruzada y el Grid para buscar los smejores parámetros del modelo. También se incorporó OneHotEncoder y StandardScaler para transformar los nombres de los directores y escritores que son categóricos, usando ColumnTransformer y el Pipeline para juntar todo el preprocesamiento de la data así como los regularizadores que penalizarán el overfit si es que hay. Finalmente se incluyen las métricas de errores; el error cuadrático medio (MSE) y error absoluto medio (MAE). 

Luego de configurar las librerías, se cargó el dataset hasta la fila 10, las filas desplegadas permiten ver el orden de las columnas también y su estructura como se aprecia en la imagen:





### Paso 2: EDA
### Paso 3: Feature Engineering:
### Paso 4: Feature Selection: 
### Paso 5: Control de overfitting:
### Paso 6: Testeo de Regresión Lineal:
### Paso 7: Visualización de resultados Y Métricas: Regresión lineal:
### Paso 8: Evaluación comparativa con Elastic Net:

## Resultados

## Conclusiones

## Referencias

* Fandom. (s.f.). *Categoría:Directores*. Los Simpson Wiki. [https://simpsons.fandom.com/es/wiki/Categor%C3%ADa:Directores](https://simpsons.fandom.com/es/wiki/Categor%C3%ADa:Directores)
* Fandom. (s.f.). *Categoría:Guionistas de Los Simpson*. Los Simpson Wiki. [https://simpsons.fandom.com/es/wiki/Categor%C3%ADa:Guionistas_de_Los_Simpson](https://simpsons.fandom.com/es/wiki/Categor%C3%ADa:Guionistas_de_Los_Simpson)
* IBM. (2025). *¿Qué es la multicolinealidad?* IBM Think Topics. [https://www.ibm.com/es-es/think/topics/multicollinearity](https://www.ibm.com/es-es/think/topics/multicollinearity)
* IBM. (2025). *What is data leakage in machine learning?* IBM Think Topics. [https://www.ibm.com/think/topics/data-leakage-machine-learning](https://www.ibm.com/think/topics/data-leakage-machine-learning)
* Sharma, A. (2022, 12 de julio). *Step-by-step exploratory data analysis (EDA) using Python*. Analytics Vidhya. [https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/)
* Singh, A. (2023, 22 de diciembre). *Building smarter ML pipelines with column transformers*. Medium. [https://medium.com/@abhaysingh71711/building-smarter-ml-pipelines-with-column-transformers-895904e97254](https://medium.com/@abhaysingh71711/building-smarter-ml-pipelines-with-column-transformers-895904e97254)
* Zhihu. (2023, 11 de mayo). *Respuesta a una consulta de ML* (Respuesta N.º 3002775487). Zhihu. [https://www.zhihu.com/en/answer/3002775487](https://www.zhihu.com/en/answer/3002775487)
Usa el código con precaución.





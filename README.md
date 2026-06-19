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

El proyecto parte con la preparación del entorno en Python, cargandao Pandas y NumPy para ordenar las tablas y hacer cálculos, y Matplotlib junto con Seaborn para generar los gráficos. Para la parte del entrenamiento y predicción se incluyen los elementos de la librería Scikit-Learn, además dentro de esta se incluye la validación cruzada y el Grid para buscar los smejores parámetros del modelo. También se incorporó OneHotEncoder y StandardScaler para transformar los nombres de los directores y escritores que son categóricos, usando ColumnTransformer y el Pipeline para juntar todo el preprocesamiento de la data así como los regularizadores que penalizarán el overfit si es que hay. Finalmente se incluyen las métricas de errores; el error cuadrático medio (MSE) y error absoluto medio (MAE). <br>
Luego de configurar las librerías, se cargó el dataset hasta la fila 10, las filas desplegadas permiten ver el orden de las columnas también y su estructura como se aprecia en la imagen:

![Vista inicial del dataset](head(10).png)

### Paso 2: EDA

El dataset de capítulos de Los Simpsons contiene 14 columnas las cuales son:
* id: Identificador único de cada episodio.
* title: Título original del episodio.
* description: Resumen o sinopsis del argumento del capítulo.
* original_air_date: Fecha en la que el episodio se emitió por primera vez en televisión.
* production_code: Código interno de producción utilizado por el estudio.
* directed_by: Nombre del director o directores del episodio.
* written_by: Nombre del guionista o guionistas que escribieron el episodio.
* season: Número de la temporada a la que pertenece el capítulo.
* number_in_season: Número del episodio dentro de su respectiva temporada.
* number_in_series: Número correlativo del episodio en toda la serie (ej. Episodio 1, 2, 3... hasta el final).
* us_viewers_in_millions: Cantidad de espectadores en Estados Unidos (medido en millones de personas).
* imdb_rating: Calificación promedio obtenida en la plataforma IMDb.
* tmdb_rating: Calificación promedio obtenida en la plataforma The Movie Database (TMDB).
* tmdb_vote_count: Cantidad total de votos/usuarios que calificaron el episodio en TMDB.

Con 747 filas correspondiente a 747 episodios de distintas temporadas. Al revisar la integridad del dataset mediante *data.isnull().sum()*, se confirma que los datos están prácticamente completos, registrando un único valor nulo en la columna *us_viewers_in_millions*, el cual corresponde a la audiencia en EE.UU. de un solo episodio. Por otra parte, al ejecutar *data.duplicated().sum()* el resultado es 0, lo que demuestra que no existen filas idénticas clonadas en su totalidad. Cabe destacar eso si que as repeticiones en columnas como directores, guionistas o temporadas son completamente normales ya que un mismo director trabaja en varios episodios y las temporadas tienen muchos capítulos. 

### Descripción del dataset

Siguiendo con la exploración inicial de la data, la siguiente imagen presenta la descripción detalladas de las columnas y sus estadísticos:

![Descripción del dataset](descripciondata.png)

* **count**: Muestra la cantidad de datos válidos que hay. La mayoría de las columnas dicen 747.000000 (747 episodios), pero us_viewers_in_millions dice 746.000000. Significa que hay 1 episodio al que le falta el dato de audiencia, tal como se vio en el estudio de datos nulos y su cantidad. 

* **mean**: se puede ver el promedio histórico de la serie. La serie promedia 17.4 temporadas, actualmente va en la temporada 37 con más de 800 episodios pero al momento de crear este dataset se consideraron solo 747. Además tiene una sintonía histórica promedio de 10.38 millones de espectadores por capítulo y la "nota" promedio en IMDb es de 7.15 dentro del rango de 1 al 10.

* **std**: la desviación estándar muestra que la audiencia es de 6.96, lo que significa que la sintonía ha cambiado drásticamente entre temporadas; alguinos capítulos con muchísima gente y otros con muy poca.

* **min (Mínimo)**: es el punto más bajo registrado con el peor capítulo de Los Simpson teniendo un 4.0 en IMDb. Y el capítulo menos visto tuvo apenas 0.77 millones de espectadores.

* **max (Máximo)**: la serie al momento de la creación de este dataset tiene un máximo de 34 temporadas, su capítulo más visto tuvo 33.6 millones de espectadores y el capítulo mejor evaluado de la historia llegó a 9.3 en IMDb.

* **Respecto a los percentiles de 25%, 50% y 75%**: el análisis del imdb_rating muestra que la calidad de la serie se concentra en rangos bien marcados. El 25% de los episodios tiene una nota de 6.60 o menos, lo que representa las etapas con menor desempeño. Por otro lado, la amediana es la mitad del show y se ubica exactamente en 7.00; al ser una nota más baja que el promedio general de 7.15, se demuestra que un grupo selecto de capítulos clásicos y muy exitosos empuja la media hacia arriba. Finalmente, el 75% de la serie alcanza como máximo un 7.70, lo que significa que solo un exclusivo 25% de los episodios logra superar esa barrera.

A continuación, se estudiaron las variables categóricas del dataset de directores y escritores, desplegando la cantidad exacta de episodios dirigidos:

| Director | Cantidad de Episodios |
| :--- | :---: |
| Mark Kirkland | 83 |
| Steven Dean Moore | 82 |
| Bob Anderson | 64 |
| Matthew Nastuk | 57 |
| Mike Frank Polcino | 39 |
| Jim Reardon | 35 |
| Chris Clements | 32 |
| Rob Oliver | 32 |
| Nancy Kruse | 26 |
| Wes Archer | 25 |

Y escritos por cada persona para identificar a los realizadores más frecuentes del programa:

| Guionista | Cantidad de Episodios |
| :--- | :---: |
| John Swartzwelder | 55 |
| Joel H. Cohen | 32 |
| Tim Long | 27 |
| J. Stewart Burns | 25 |
| Michael Price | 25 |
| John Frink | 24 |
| Matt Selman | 24 |
| Jeff Westbrook | 22 |
| Jon Vitti | 22 |
| Carolyn Omine | 19 |

Para entender cómo se relacionan la recepción crítica (IMDB y TMDB) y la popularidad de los episodios, se generaron cuatro boxplots combinando distintas variables del dataset:

* Boxplot 1: Calificaciones de IMDb.
* Boxplot 2: Calificaciones de TMDB.
* Boxplot 3: Calificaciones de Audiencia en millones.
* Boxplot 4: Calificaciones de Distribución de votos.

![Descripción del dataset](boxplot.png)

Se realizaron boxplots únicamente para las variables numéricas continuas que aportaban algo sustancial (imdb_rating, tmdb_rating, us_viewers_in_millions y tmdb_vote_count), ya que son las más adecuadas para analizar su distribución y detectar posibles valores atípicos. Las variables discretas, como el número de temporada o de episodio, no aportan información relevante mediante este tipo de gráfico.

1. Distribución de Calificaciones IMDb:
Este boxplot analiza el comportamiento de las notas en la plataforma IMDb, donde califican expertos en el tema y profesionales de la industria. El 50% central de los episodios se concentra de dos maneras; parte en una nota de 6.6 aprox, siendo el 25% con peor puntaje y llega hasta el 7.7 aprox que corresponde al 75%. La línea del medio, que es la mediana, queda exactamente en 7.0; como está un poco por debajo del promedio general de 7.15, se nota a simple vista que los capítulos clásicos hacen la balanza hacia arriba. Por el lado de los mejores episodios, el gráfico sube hasta la nota máxima de 9.3. En cambio, para los valores más bajos el límite normal cae hasta los 5.0 puntos y, debajo de eso, quedan tres capítulos sueltos como valores atípicos (en 4.4 y dos en 4.0), que son oficialmente los peores evaluados de toda la serie.

2. Distribución de Calificaciones TMDB: 
Este boxplot de caja para las notas de TMDB nos muestra que esta comunidad es un poco más exigente, siendo calificada por los usuarios y no por los directores o especialistas en el tema. El 50% central de los capítulos se mueve en un rango más bajo que el de IMDb: empieza en 5.8 y llega solo hasta 7.0 aaprox. La línea del medio, es decir, la mediana corta justo en 6.3, lo que confirma que en TMDB las calificaciones suelen ser más bajas a nivel general. Por la parte de arriba, los mejores episodios evaluados llegan hasta una nota de 8.6. En la parte de menores calificaciones el límite normal cae hasta los 4.0 puntos, pero por debajo de eso aparecen dos valores atípicos; un capítulo en los 3.3 puntos aprox y un episodio con una nota de 0.0, lo que podría ser un error de registro.

3. Distribución de Audiencia en EE.UU:
Este boxplot muestra los datos de la audiencia en televisión, y de todos los gráficos del análisis, este es el que tiene los datos más dispersos. La mitad central de la serie comienza en 4.75 millones aprox de espectadores (el 25% con menos público) y sube hasta los 14.8 millones (el 75%) aprox. La línea del medio, que es la mediana, queda aproximadamente en los 9 millones. Por el lado de abajo, el límite roza el mínimo del show, que es de apenas 0.77 millones en las temporadas más nuevas. Por arriba, el límite llega hasta los 30 millones, pero pasando esa línea aparecen tres capítulos sueltos como valores atípicos; siendo probablemente los capítulos mejor calificados de todos los años.

4. Distribución de votos TMDB:
El boxplot muestra que la mayor parte de los episodios concentra entre aproximadamente 14 y 23 mil votos, con una mediana cercana a 17 mil votos. Además, se observan varios valores atípicos superiores, alcanzando más de 60 mil votos, lo que indica que algunos episodios recibieron una cantidad de votaciones considerablemente mayor que el resto, probablemente debido a su mayor popularidad por parte d elos fanáticos.

Siguiendo con el análisis para contrastar la opinión de la audiencia con el fenómeno cultural de la serie, se graficó una línea de tendencia del puntaje promedio en IMDb a lo largo de todas las temporadas. Esta visualización busca identificar con precisión matemática el periodo conocido como la "época dorada" de Los Simpson y determinar el punto exacto en el que las temporadas empezaron a perder audiencia.

![Línea de audiencia vs temporada](goldenera.png)

Así, el gráfico de arriba muestra el promedio de rating por temporada para identificar la época dorada de la serie; en el eje X está la temporada vs el promedio de imdb_rating. La imagen del gráfico revela una tendencia marcada respecto a la calificación promedio de la serie a lo largo de sus 34 temporadas hasta esa fecha.

La evolución de la serie se puede dividir en tres etapas muy marcadas si se considera un promedio de 8 puntos en IMDb como éxito. Todo empieza con la denominada "Época Dorada" por los fans desde aproximadamente las Temporadas 3 a 8, donde la calidad es desde la primera temporada hasta alcanzar una excelencia absoluta entre la 4 y la 7, logrando un pico histórico en la temporada 7 con un promedio de 8.3 aprox; esta era coincide con los años dorados de escritores y directores de culto conocidos por los fans. Luego viene la fase de comienzo del declive desde las Temporadas 9 a 12, donde los datos muestran una caaída a los 7.5 puntos en la temporada 10 y se estanca en un 7.3 en la 11, marcando el fin definitivo de la era clásica. Finalmente, la serie entra en un declive final desde las Temporadas 13 a 34 creo que debido al desgaste del show, manteniéndose en un rango bajo entre el 7.0 y el 6.5, y llegando a lo más bajo en la temporada 30 con una nota promedio de 6.2 aprox, posteriormente la serie repunta débilmente.

Siguiendo con el análisis de calidad, se generó un gráfico para identificar a los directores mejor evaluados en IMDb y descubrir quiénes están detrás de los capítulos con las notas más altas de la serie.

![Directores mejor evaluados](top_directores.png)

El análisis demuestra que la dirección es clave para el éxito de Los Simpsons, pero una mayor cantidad de episodios dirigidos no asegura una mejor calificación. Los realizadores que más capítulos acumulan en la historia del show, como Mark Kirkland con 83 episodios, Steven Dean Moore con 82, Bob Anderson con 64 o Matthew Nastuk con 57, quedan fuera de la lista de los mejor puntuados. En contraste, las notas de excelencia sobre los 8.0 puntos están concentradas en los directores de la época dorada, como Jeffrey Lynch, Rich Moore, Jim Reardon, Wes Archer, Carlos Baeza y Susie Dietter, quienes definieron la identidad visual más recordada de la serie. Por su parte, David Silverman se mantiene en un punto intermedio como un director histórico de alto volumen que promedia un 7.8, superando por mucho la media general de la serie que se ubica en 7.15. Esto comprueba que las calificaciones más altas dependen de su época en el show, más que de la cantidad de trabajo producido.

De igual manera se realizó algo similiar con los escritores:

![Escritores mejor evaluados](top_escritores.png)

Bill Oakley y Josh Weinstein lideran el análisis de guionistas con un promedio de 8.25, seguidos por Mike Scully y el dúo de Kogen & Wolodarsky, mientras John Swartzwelder destaca por mantener un alto nivel (7.9) a pesar de ser el más frecuente. En contraste nuevamente, una alta producción de episodios no garantiza calidad, evidenciado por el promedio de 7.15 de Al Jean, marcando el fracaso de la era moderna.

Finalmente se realizó un scatterplot que revela una relación clara entre la audiencia y la calificación de los episodios, mostrando que ambas variables avanzan juntas en los rangos bajos y medios: a medida que el público crece de 1 a 15 millones, las notas tienden a subir de forma notable. Esto explica por qué los capítulos modernos se concentran en la parte inferior izquierda con bajas audiencias y notas de 6.0 a 7.0, mientras que la masa de la era clásica se posiciona en la zona central superior, combinando entre 12 y 20 millones de espectadores con excelentes promedios sobre los 8.0 puntos. Sin embargo, al superar el umbral de los 25 millones de personas, esta tendencia lineal se rompe y ocurre que estos episodios se estabilizan en un rango de 7.5 a 8.5. Este comportamiento demuestra que un éxito masivo en sintonía no asegura una mejor recepción por parte de la crítica, independizándose el volumen de espectadores de la calidad percibida en los picos más altos de popularidad.

![Escritores mejor evaluados](scatterplotEDA.png)

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





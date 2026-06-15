# Proyecto-1-IA-Los-Simpsons
Desarrollo de Proyecto 1 de Introducción a la Inteligencia Artificial 

# Definición del problema

## Claridad y Relevancia

El proyecto está basado en la serie norteamericana "Los Simpsons" que a lo largo de sus 39 años ha experimentado éxitos y grandes valores de sintonía como en la valoración de su audiencia. El problema consiste en analizar qué características de los episodios están asociadas a una mejor recepción por parte del público y desarrollar un modelo capaz de predecir la calificación IMDb de un episodio a partir de variables relacionadas con su producción y desempeño. Entre estas variables se consideran la temporada, el número de episodio, la audiencia en millones de espectadores, los responsables de dirección y escritura. Permitiendo todo esto comprender qué factores influyen directamente en el éxito prolongado del show.

# Plan de acción

## Descripción del dataset

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

Para poder explorar de manera más específica los datos se realizarán análisis exploratorios y correlaciones.

## Modelo(s) seleccionado(s) y estrategia de evaluación claramente explicados

Se seleccionó un modelo de Regresión Lineal, utilizando como variable objetivo la calificación IMDb (imdb_rating). El modelo buscará estimar la puntuación de un episodio a partir de las características disponibles en el conjunto de datos, identificando las variables que presentan mayor influencia sobre la valoración de los espectadores.
Para evaluar su desempeño se utilizará validación cruzada K-Fold con k=5, técnica que permite obtener una estimación más robusta de la capacidad predictiva del modelo (mejor que dividir en 80/20 para evitar perder información evaluando el modelo con las 750 filas completas en diferentes turnos). En cada iteración, cuatro particiones serán utilizadas para entrenamiento y una para validación, repitiendo el proceso hasta que todas las observaciones hayan sido evaluadas. El rendimiento se analizará mediante las métricas R², MAE y RMSE.

## Justifiación de modelo: ventajas, limitaciones y pertinencia.

Se seleccionó la Regresión Lineal debido a que la variable objetivo del estudio, imdb_rating, corresponde a una variable numérica continua. Este tipo de modelo permite analizar y cuantificar la relación existente entre una variable dependiente y múltiples variables explicativas, por lo que resulta adecuado para estimar la valoración de un episodio a partir de sus características. 

Entre sus principales ventajas se encuentran su simplicidad de implementación y facilidad de interpretación. Además, permite identificar qué variables tienen una mayor influencia sobre la calificación IMDb mediante el análisis de los coeficientes asociados a cada predictor. Como limitación, la regresión lineal supone que existe una relación lineal entre las variables analizadas, por lo que puede que no llegue a representar correctamente patrones más complejos. Sin embargo, este modelo resulta pertinente para el problema planteado, ya que permite predecir una variable numérica como la calificación IMDb e identificar qué factores influyen en la valoración de los episodios. Además, destaca por ser un modelo simple, fácil de interpretar y adecuado para un análisis exploratorio inicial.

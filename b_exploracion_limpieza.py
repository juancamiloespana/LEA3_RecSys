import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import a_funciones as fn


import os  ### para ver y cambiar directorio de trabajo



os.getcwd()
os.chdir('d:\\Docencia\\Analítica3\\marketing')


###### para ejecutar sql y conectarse a bd ###

conn=sql.connect('db_books2')
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

############ cargar tablas ####


books= pd.read_sql('select * from books', conn)
book_ratings = pd.read_sql('select * from book_ratings', conn)
users=pd.read_sql('select * from users', conn)

#####Exploración inicial #####

### Identificar campos de cruce y verificar que estén en mismo formato ####
### verificar duplicados


books.info()
books.head()
books.duplicated().sum() 

book_ratings.info()
book_ratings.head()
book_ratings.duplicated().sum()

users.info()
users.head()
users.duplicated().sum()


##### Descripción base de ratings


cr=pd.read_sql(""" select 
                          "Book-Rating" as rating, 
                          count(*) as conteo 
                          from book_ratings
                          group by "Book-Rating"
                          """, conn)
###Nombres de columnas con numeros o guiones se deben poner en doble comilla para que se reconozcan


data  = go.Bar( x=cr.rating,y=cr.conteo, text=cr.conteo, textposition="outside")
Layout=go.Layout(title="Count of ratings",xaxis={'title':'Rating'},yaxis={'title':'Count'})
go.Figure(data,Layout)

### los que están en 0 fueron leídos pero no calificados

rating_users=pd.read_sql(''' select "User-Id" as user_id,
                         count(*) as cnt_rat
                         from book_ratings
                         group by "User-Id"
                         order by cnt_rat desc
                         ''',conn )

data  = go.Scatter(x = rating_users.index, y= rating_users.cnt_rat)
Layout= go.Layout(title="Ratings given per user",xaxis={'title':'User Count'}, yaxis={'title':'Ratings'})
go.Figure(data, Layout)  

###Son pocos usuarios los que calificaron varios libros

rating_books=pd.read_sql(''' select ISBN ,
                         count(*) as cnt_rat
                         from book_ratings
                         group by "ISBN"
                         order by cnt_rat desc
                         ''',conn )


data  = go.Scatter(x = rating_books.index, y= rating_books.cnt_rat)
Layout= go.Layout(title="Ratings received per book",xaxis={'title':'Book Count'}, yaxis={'title':'Ratings'})
go.Figure(data, Layout)    


####Filtrar libros que no tengan más de 10 calificaciones y usuarios que no tengan más de 10 libros calificados

fn.ejecutar_sql('preprocesamientos.sql', cur)

pd.read_sql('select count(*) from users_final', conn)
pd.read_sql('select count(*) from books_final', conn)
pd.read_sql('select count(*) from ratings_final', conn)
pd.read_sql('select count(*) from full_ratings', conn)

ratings=pd.read_sql('select * from full_ratings',conn)
ratings.duplicated().sum() ## al cruzar tablas a veces se duplican registros
ratings.info()


#### Explorar tabla completa ---Ejercicios ----

### Libro publicado en 2004 más leido: 0,3 
### Libro mejor calificado: 0,2
### Autor más leido 0,2
### Editorial más leida 0,2
### Libro más mal calificado 0,2
### qué películas  y años de publicación, tienen lectores con edad promedio de 52 años



import pandas as pd
import sqlite3 as sql
df=pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/RecSys/main/full_ratings.csv')
conn=sql.connect('db_books_ejercicio')
cur=conn.cursor()
df.to_sql('full_ratings',conn)

pd.read_sql('''
            ''', conn)
import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
import a_funciones as fn


###### para ejecutar sql y conectarse a bd ###

## crear copia de db_books datos originales, nombrarla books2 y procesar books2

conn=sql.connect('db_books2') ### crear cuando no existe el nombre de cd  y para conectarse cuando sí existe.
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos

### para verificar las tablas que hay disponibles
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()

#######
############ traer tabla de BD a python ####


books= pd.read_sql("""select *  from books""", conn)
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

###calcular la distribución de calificaciones
cr=pd.read_sql(""" select 
                          "Book-Rating" as rating, 
                          count(*) as conteo 
                          from book_ratings
                          group by "Book-Rating"
                          order by conteo desc""", conn)
###Nombres de columnas con numeros o guiones se deben poner en doble comilla para que se reconozcan


data  = go.Bar( x=cr.rating,y=cr.conteo, text=cr.conteo, textposition="outside")
Layout=go.Layout(title="Count of ratings",xaxis={'title':'Rating'},yaxis={'title':'Count'})
go.Figure(data,Layout)

### los que están en 0 fueron leídos pero no calificados
#### Se conoce como calificación implicita, consume producto pero no da una calificacion


### calcular cada usuario cuátos libros calificó
rating_users=pd.read_sql(''' select "User-Id" as user_id,
                         count(*) as cnt_rat
                         from book_ratings
                         group by "User-Id"
                         order by cnt_rat asc
                         ''',conn )

fig  = px.histogram(rating_users, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show() 


rating_users.describe()
### la mayoria de usarios tiene pocos libros calificados, pero los que más tiene tiene muchos

#### filtrar usuarios con más de 50 libros calificados (para tener calificaion confiable) y los que tienen mas de mill porque pueden ser no razonables
rating_users2=pd.read_sql(''' select "User-Id" as user_id,
                         count(*) as cnt_rat
                         from book_ratings
                         group by "User-Id"
                         having cnt_rat >=50 and cnt_rat <=1000
                         order by cnt_rat asc
                         ''',conn )

### ver distribucion despues de filtros,ahora se ve mas razonables
rating_users2.describe()


### graficar distribucion despues de filtrar datos
fig  = px.histogram(rating_users2, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones por usario')
fig.show() 


#### verificar cuantas calificaciones tiene cada libro
rating_books=pd.read_sql(''' select ISBN ,
                         count(*) as cnt_rat
                         from book_ratings
                         group by "ISBN"
                         order by cnt_rat desc
                         ''',conn )

### analizar distribucion de calificaciones por libro
rating_books.describe()

### graficar distribucion

fig  = px.histogram(rating_books, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones para cada libro')
fig.show()  
####Filtrar libros que no tengan más de 50 calificaciones y usuarios que no tengan más de 10 libros calificados
rating_books2=pd.read_sql(''' select ISBN ,
                         count(*) as cnt_rat
                         from book_ratings
                         group by "ISBN"
                         having cnt_rat>=50
                         order by cnt_rat desc
                         ''',conn )

rating_books2.describe()
fig  = px.histogram(rating_books2, x= 'cnt_rat', title= 'Hist frecuencia de numero de calificaciones para cada libro')
fig.show()

###########
fn.ejecutar_sql('preprocesamientos.sql', cur)

cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()


### verficar tamaño de tablas con filtros ####

## users
pd.read_sql('select count(*) from users', conn)
pd.read_sql('select count(*) from users_final', conn)

####books

pd.read_sql('select count(*) from books', conn)
pd.read_sql('select count(*) from books_final', conn)

##ratings
pd.read_sql('select count(*) from book_ratings', conn)
pd.read_sql('select count(*) from ratings_final', conn)

## 3 tablas cruzadas ###
pd.read_sql('select count(*) from full_ratings', conn)

ratings=pd.read_sql('select * from full_ratings',conn)
ratings.duplicated().sum() ## al cruzar tablas a veces se duplican registros
ratings.info()
ratings.head(10)

### tabla de full ratings se utilizara para modelos #####

##### recomendaciones basado en popularidad ######

#### mejores calificadas que tengan calificacion
pd.read_sql("""select book_title, 
            avg(book_rating) as avg_rat,
            count(*) as read_num
            from full_ratings
            where book_rating<>0
            group by book_title
            order by avg_rat desc
            limit 10
            
            """, conn)

#### los mas leidos con promedio de los que calficaron ###
pd.read_sql("""select book_title, 
            avg(iif(book_rating = 0, Null, book_rating)) as avg_rat,
            count(*) as read_num
            from full_ratings
            group by book_title
            order by read_num desc
            """, conn)


#### los mejores calificados por año publicacion ###
pd.read_sql("""select year_pub, book_title, 
            avg(iif(book_rating = 0, Null, book_rating)) as avg_rat,
            count(iif(book_rating = 0, Null, book_rating)) as rat_numb,
            count(*) as read_num
            from full_ratings
            group by  year_pub, book_title
            order by year_pub desc, avg_rat desc
            """, conn)



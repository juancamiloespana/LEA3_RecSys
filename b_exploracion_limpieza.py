import numpy as np
import pandas as pd
import sqlite3 as sql
import plotly.graph_objs as go ### para gráficos
import plotly.express as px
import a_funciones as fn

###pip install  pysqlite3

###### para ejecutar sql y conectarse a bd ###

## crear copia de db_books datos originales, nombrarla books2 y procesar books2

conn=sql.connect('data\\db_books2') ### crear cuando no existe el nombre de cd  y para conectarse cuando sí existe.
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos



#### para consultar datos ######## con cur

cur.execute("select * from books")
cur.fetchall()

##### consultar trayendo para pandas ###
df_books=pd.read_sql("select * from books", conn)


#### para ejecutar algunas consultas

cur.execute("drop table if exists books3")
cur.execute(""" create table books3 as select * from 
            books where "Year-Of-Publication" = '2002' """)

pd.read_sql("select * from books3", conn)


#### para llevar de pandas a BD
df_books.to_sql("books3", conn, if_exists='replace')
###conn.close()para cerrar conexión


### para verificar las tablas que hay disponibles
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()


#######
############ traer tabla de BD a python ####


books= pd.read_sql("""select *  
                   from books 
                   """, conn)

book_ratings = pd.read_sql('select * from book_ratings', conn)

users=pd.read_sql('select * from users', conn)

cur.execute(" drop table books3")

cur.execute(""" create table books3 
            as select *, cast("Year-Of-Publication" as int) as year_pub 
            from books 
            where "Year-Of-Publication"= "2002"  """)

books3= pd.read_sql("""select "Book-Author" as author, avg(year_pub) as prom_anho  
                   from books3 
                   group by author
                   """, conn)

books3.info()



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
### la mayoria de usarios tiene pocos libros calificados, pero los que más tienen muchos

#### excluir usuarios con menos de 50 libros calificados (para tener calificaion confiable) y los que tienen mas de mil porque pueden ser no razonables

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
####Excluir libros que no tengan más de 50 calificaciones 
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


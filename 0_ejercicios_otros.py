##########################################################################################
########################## Introduccion a sql ###########################################
##########################################################################################

import os  ### para ver y cambiar directorio de trabajo
import pandas as pd
import sqlite3 as sql ### para conectarse a BD


os.getcwd() ## ver directorio actual
os.chdir('d:\\cod\\marketing') ### cambiar directorio a ruta específica

##### conectarse a BD #######
conn= sql.connect('db_movies')
cur=conn.cursor()

### para ver las tablas que hay en la base de datos
cur.execute("select name from sqlite_master where type='table' ")
cur.fetchall()

######1 ejercicios sql con base de movies(estudiantes) ###

pd.read_sql("""select genres, count(*) from movies group by genres""", conn)
pd.read_sql("""select * from ratings""", conn)
pd.read_sql("""select * from movies""", conn)
pd.read_sql("""select count(*) from movies""", conn)

pd.read_sql("""select count(distinct userId) from ratings""", conn)
pd.read_sql("""select genres, count(*) as cnt 
            from movies 
            group by genres 
            order by cnt desc """, conn).head(10)

pd.read_sql("""select genres, count(*) as cnt 
            from movies 
            group by genres 
            order by cnt desc """, conn).head(10)


pd.read_sql("""select userId, avg(rating)
            from ratings
            group by userId order by userId asc""", conn)


pd.read_sql("""select a.title, count(b.rating) as cnt
            from movies a left join ratings b on a.movieId=b.movieId 
            group by a.title having cnt=0 order by cnt asc """, conn)


##########################################################################################
#########################################################################################
### ejercicio sql con base de datos de libros #############################################
###############################################################################################

df=pd.read_csv('https://raw.githubusercontent.com/juancamiloespana/RecSys/main/full_ratings.csv')

#### Explorar tabla completa ---Ejercicios ----
### leer tabla de github (full_ratings)
### crear una base de datos vacía y llevar la tabla fullratings a la base de datos
### Libro publicado en 2004 más leido: 
### Libro mejor calificado: 0,2
### Autor más leido 0,2
### Editorial más leida 0,2
### Libro más mal calificado 0,2
### qué películas  y años de publicación, tienen lectores con edad promedio de 52 años



###################################################################################
###################################################################################
####codigo para separar género #############################################
##########################################################################################


import pandas as pd
import sqlite3 as sql ### para conectarse a BD
from mlxtend.preprocessing import TransactionEncoder

conn= sql.connect('db_movies')
cur=conn.cursor()

movies=pd.read_sql("""select * from movies""", conn)
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)



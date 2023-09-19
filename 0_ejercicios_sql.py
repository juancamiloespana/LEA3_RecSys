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

###1
pd.read_sql("select * from ratings", conn)


pd.read_sql("""select count(*) from movies""", conn)

###2
pd.read_sql("""select count(distinct userId) from ratings""", conn)


####3
pd.read_sql("""select movieId, avg(rating)
            from ratings
            where movieId=1
            group by movieId order by userId asc""", conn)

####4
pd.read_sql("""select a.title, count(b.rating) as cnt
            from movies a left join ratings b on a.movieId=b.movieId 
            group by a.title where b.rating is null order by cnt asc """, conn)

####5
pd.read_sql("""select a.title, count(b.rating) as cnt
            from movies a left join ratings b on a.movieId=b.movieId 
            group by a.title having cnt=1 order by cnt asc """, conn)

####6 
pd.read_sql("""select genres, count(*) as cnt
            from movies 
            group by genres 
            order by cnt desc limit 8,1 """, conn)

pd.read_sql("""with t1 as (select genres, count(*) as cnt 
            from movies 
            group by genres 
            order by cnt desc limit 9) select * from t1 order by cnt asc limit 1 """, conn)


pd.read_sql("""select userId, avg(rating)
            from ratings
            group by userId order by userId asc""", conn)



###################################################################################
###################################################################################
####codigo para separar género #############################################
##########################################################################################


import pandas as pd
import sqlite3 as sql ### para conectarse a BD
from mlxtend.preprocessing import TransactionEncoder

conn= sql.connect('data\\db_movies')
cur=conn.cursor()

movies=pd.read_sql("""select * from movies""", conn)
genres=movies['genres'].str.split('|')
te = TransactionEncoder()
genres = te.fit_transform(genres)
genres = pd.DataFrame(genres, columns = te.columns_)
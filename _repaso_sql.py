import numpy as np
import pandas as pd
import sqlite3 as sql


###pip install  pysqlite3

###### para ejecutar sql y conectarse a bd ###

## crear copia de db_books datos originales, nombrarla books2 y procesar books2

conn=sql.connect('data\\db_books2') ### crear cuando no existe el nombre de cd  y para conectarse cuando sí existe.
cur=conn.cursor() ###para funciones que ejecutan sql en base de datos


### para verificar las tablas que hay disponibles
cur.execute("SELECT name FROM sqlite_master where type='table' ")
cur.fetchall()



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
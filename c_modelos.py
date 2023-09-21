import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido
import joblib
#### conectar_base_de_Datos

conn=sql.connect('data\\db_books2')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()


######################################################################
################## 1. sistemas basados en popularidad ###############
#####################################################################


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
            order by year_pub desc, avg_rat desc limit 20
            """, conn)




#######################################################################
######## 2.1 Sistema de recomendación basado en contenido un solo producto - Manual ########
#######################################################################

books=pd.read_sql('select * from books_final', conn )

books.info()
books['year_pub']=books.year_pub.astype('int')
books.info()

##### escalar para que año esté en el mismo rango ###

sc=MinMaxScaler()
books[["year_sc"]]=sc.fit_transform(books[['year_pub']])



## eliminar filas que no se van a utilizar ###

books_dum1=books.drop(columns=['isbn','i_url','year_pub','book_title'])

#### convertir a dummies

books_dum1['book_author'].nunique()
books_dum1['publisher'].nunique()

col_dum=['book_author','publisher']
books_dum2=pd.get_dummies(books_dum1,columns=col_dum)
books_dum2.shape

joblib.dump(books_dum2,"salidas\\books_dum2") ### para utilizar en segundos modelos



###### libros recomendadas ejemplo para un libro#####

libro='The Testament'
ind_libro=books[books['book_title']==libro].index.values.astype(int)[0]
similar_books=books_dum2.corrwith(books_dum2.iloc[ind_libro,:],axis=1)
similar_books=similar_books.sort_values(ascending=False)
top_similar_books=similar_books.to_frame(name="correlación").iloc[0:11,] ### el 11 es número de libros recomendados
top_similar_books['book_title']=books["book_title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    


#### libros recomendados ejemplo para visualización todos los libros

def recomendacion(libro = list(books['book_title'])):
     
    ind_libro=books[books['book_title']==libro].index.values.astype(int)[0]   #### obtener indice de libro seleccionado de lista
    similar_books = books_dum2.corrwith(books_dum2.iloc[ind_libro,:],axis=1) ## correlación entre libro seleccionado y todos los otros
    similar_books = similar_books.sort_values(ascending=False) #### ordenar correlaciones
    top_similar_books=similar_books.to_frame(name="correlación").iloc[0:11,] ### el 11 es número de libros recomendados
    top_similar_books['book_title']=books["book_title"] ### agregaro los nombres (como tiene mismo indice no se debe cruzar)
    
    return top_similar_books


print(interact(recomendacion))


##############################################################################################
#### 2.1 Sistema de recomendación basado en contenido KNN un solo producto visto #################
##############################################################################################

##### ### entrenar modelo #####

## el coseno de un angulo entre dos vectores es 1 cuando son perpendiculares y 0 cuando son paralelos(indicando que son muy similar324e-06	3.336112e-01	3.336665e-01	3.336665e-es)
model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(books_dum2)
dist, idlist = model.kneighbors(books_dum2)

distancias=pd.DataFrame(dist) ## devuelve un ranking de la distancias más cercanas para cada fila(libro)
id_list=pd.DataFrame(idlist) ## para saber esas distancias a que item corresponde


####ejemplo para un libro
book_list_name = []
book_name='Violets Are Blue'
book_id = books[books['book_title'] == book_name].index ### extraer el indice del libro
book_id = book_id[0] ## si encuentra varios solo guarde uno
for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].book_title) ### agrega el nombre de cada una de los id recomendados

book_list_name




def BookRecommender(book_name = list(books['book_title'].value_counts().index)):
    book_list_name = []
    book_id = books[books['book_title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].book_title)
    return book_list_name


print(interact(BookRecommender))





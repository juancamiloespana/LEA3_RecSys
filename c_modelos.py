import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo
from sklearn import neighbors ### basado en contenido un solo producto consumido

#### conectar_base_de_Datos

conn=sql.connect('db_books2')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()

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


###### mostrar películas recomendadas #####

libro='The Testament'
ind_libro=books[books['book_title']==libro].index.values.astype(int)[0]
correlaciones=books_dum2.corrwith(books_dum2.iloc[ind_libro,:],axis=1)
correlaciones.sort_values(ascending=False)


def recomendacion(libro = list(books['book_title'])):
     
    ind_libro=books[books['book_title']==libro].index.values.astype(int)[0]   
    similar_books = books_dum2.corrwith(books_dum2.iloc[ind_libro,:],axis=1)
    similar_books = similar_books.sort_values(ascending=False)
    top_similar_books=similar_books.to_frame(name="correlación").iloc[0:11,]
    top_similar_books['book_title']=books["book_title"]
    
    return top_similar_books


print(interact(recomendacion))


##############################################################################################
#### 2.1 Sistema de recomendación basado en contenido KNN un solo producto visto #################
##############################################################################################

##### ### entrenar modelo #####

model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(books_dum2)
dist, idlist = model.kneighbors(books_dum2)

distancias=pd.DataFrame(dist)
id_list=pd.DataFrame(idlist)

book_name='Violets Are Blue'

def BookRecommender(book_name = list(books['book_title'].value_counts().index)):
    book_list_name = []
    book_id = books[books['book_title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].book_title)
    return book_list_name



print(interact(BookRecommender))





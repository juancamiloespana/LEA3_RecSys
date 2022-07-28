import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo

####Paquete para sistema basado en contenido ####
from sklearn import neighbors


#############################################
#### conectar_base_de_Datos#################
############################################

conn=sql.connect('db_books2')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()


#######################################################################
#### Sistema de recomendación basado en contenido KNN #################
#######################################################################

books=pd.read_sql('select * from books_final', conn )


books['year_pub']=books.year_pub.astype('int')

##### escalar para que año esté en el mismo rango ###

sc=MinMaxScaler()
books[["year_sc"]]=sc.fit_transform(books[['year_pub']])

## eliminar filas que no se van a utilizar ###

books_dum1=books.drop(columns=['isbn','i_url','year_pub','book_title'])
books_dum1['book_author'].nunique()
books_dum1['publisher'].nunique()

col_dum=['book_author','publisher']
books_dum2=pd.get_dummies(books_dum1,columns=col_dum)
books_dum2.shape

##### ### entrenar modelo #####

model = neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
model.fit(books_dum2)
dist, idlist = model.kneighbors(books_dum2)

distancias=pd.DataFrame(dist)
id_list=pd.DataFrame(idlist)




def BookRecommender(book_name = list(books['book_title'].value_counts().index)):
    book_list_name = []
    book_id = books[books['book_title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(books.loc[newid].book_title)
    return book_list_name



print(interact(BookRecommender))


#######################################################################
#### Sistema de recomendación basado en contenido KNN #################
#### Con base en todo lo visto por el usuario #######################
#######################################################################

usuarios=pd.read_sql('select distinct (user_id) as user_id from ratings_final',conn)


user_id=31226

def recomendar(user_id=list(usuarios['user_id'].value_counts().index)):
    
    ratings=pd.read_sql('select *from ratings_final where user_id=:user',conn, params={'user':user_id})
    l_books_r=ratings['isbn'].to_numpy()
    books_dum2[['isbn','book_title']]=books[['isbn','book_title']]
    books_r=books_dum2[books_dum2['isbn'].isin(l_books_r)]
    books_r=books_r.drop(columns=['isbn','book_title'])
    books_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    centroide=books_r.groupby("indice").mean()
    
    
    books_nr=books_dum2[~books_dum2['isbn'].isin(l_books_r)]
    books_nr=books_nr.drop(columns=['isbn','book_title'])
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(books_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0]
    recomend_b=books.loc[ids][['book_title','isbn']]
    leidos=books[books['isbn'].isin(l_books_r)][['book_title','isbn']]
    
    return recomend_b

recomendar(52853)


print(interact(recomendar))
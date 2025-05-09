import numpy as np
import pandas as pd
import sqlite3 as sql
import a_funciones as fn ## para procesamiento
import openpyxl


####Paquete para sistema basado en contenido ####
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

import logging
from tqdm import tqdm

# Conffiguración del logg
logging.basicConfig(
    filename='D:\\GDRIVE_JCE\\Mi unidad\\cod\\LEA3_RecSys\\salidas\\reco\\script_log.log',  
    level=logging.INFO,       #nivel de logg, informativo solo muestra informació básica de ejecución
    format='%(asctime)s - %(levelname)s - %(message)s'  ## formato log, hora, nievel(error, informativo, advertencia), mensaje personalizado cuando se llama logg
)


def preprocesar(conn=None, cur=None):


    
    ######## convertir datos crudos a bases filtradas por usuarios que tengan cierto número de calificaciones
    fn.ejecutar_sql('D:\\GDRIVE_JCE\\Mi unidad\\cod\\LEA3_RecSys\\preprocesamientos.sql', cur)
    
    log_mes='Ejecución de SQL para filtrar libros y ratings completada.'
    logging.info(log_mes)
    print(log_mes)
    

    ##### llevar datos que cambian constantemente a python ######
    books=pd.read_sql('select * from books_final', conn )
    ratings=pd.read_sql('select * from ratings_final', conn)
    usuarios=pd.read_sql('select distinct (user_id) as user_id from ratings_final',conn)

    
    #### transformación de datos crudos - Preprocesamiento ################
    books['year_pub']=books.year_pub.astype('int')

    ##### escalar para que año esté en el mismo rango ###
    sc=MinMaxScaler()
    books[["year_sc"]]=sc.fit_transform(books[['year_pub']])

    ## eliminar filas que no se van a utilizar ###
    books_dum1=books.drop(columns=['isbn','i_url','year_pub','book_title'])

    col_dum=['book_author','publisher']
    books_dum2=pd.get_dummies(books_dum1,columns=col_dum)

    log_mes='Preprocesamiento de datos completado.'
    logging.info(log_mes)   
    print(log_mes)
    return books_dum2,books, conn, cur

##########################################################################
###############Función para entrenar modelo por cada usuario ##########
###############Basado en contenido todo lo visto por el usuario Knn#############################
def recomendar(user_id, conn=None, cur=None, books_dum2=None, books=None):   
    
    
    
    ratings=pd.read_sql('select *from ratings_final where user_id=:user',conn, params={'user':user_id})
    l_books_r=ratings['isbn'].to_numpy()
    books_dum2[['isbn','book_title']]=books[['isbn','book_title']]
    books_r=books_dum2[books_dum2['isbn'].isin(l_books_r)]
    books_r=books_r.drop(columns=['isbn','book_title'])
    books_r["indice"]=1 ### para usar group by y que quede en formato pandas tabla de centroide
    centroide=books_r.groupby("indice").mean()

    log_mes=f'Generando recomendaciones para el usuario {user_id}.'
    logging.info(log_mes)   
    print(log_mes)


    
    
    books_nr=books_dum2[~books_dum2['isbn'].isin(l_books_r)]
    books_nr=books_nr.drop(columns=['isbn','book_title'])
    model=neighbors.NearestNeighbors(n_neighbors=11, metric='cosine')
    model.fit(books_nr)
    dist, idlist = model.kneighbors(centroide)
    
    ids=idlist[0]
    recomend_b=books.loc[ids][['book_title','isbn']]

    log_mes=f'Recomendaciones para el usuario {user_id} finalizadas'
    logging.info(log_mes)
    print(log_mes)
    
    return recomend_b


##### Generar recomendaciones para usuario lista de usuarios ####
##### No se hace para todos porque es muy pesado #############
def main(list_user):

    #### conectar_base_de_Datos#################
    conn=sql.connect('D:\\GDRIVE_JCE\\Mi unidad\\cod\\LEA3_RecSys\\data\\db_books2')
    cur=conn.cursor()
    
    log_mes='Conexión a la base de datos establecida.'
    logging.info(log_mes)       
    print(log_mes)  

    recomendaciones_todos=pd.DataFrame()

    books_dum2, books, conn, cur= preprocesar(conn, cur)
    for user_id in tqdm(list_user):
            
        recomendaciones=recomendar(user_id, conn, cur, books_dum2, books)
        recomendaciones["user_id"]=user_id
        recomendaciones.reset_index(inplace=True,drop=True)
        
        recomendaciones_todos=pd.concat([recomendaciones_todos, recomendaciones])

    recomendaciones_todos.to_excel('D:\\GDRIVE_JCE\\Mi unidad\\cod\\LEA3_RecSys\\salidas\\reco\\recomendaciones.xlsx')
    recomendaciones_todos.to_csv('D:\\GDRIVE_JCE\\Mi unidad\\cod\\LEA3_RecSys\\salidas\\reco\\recomendaciones.csv')
    
    log_mes='Recomendaciones generadas y guardadas en Excel y CSV.' 
    logging.info(log_mes)
    print(log_mes)

if __name__=="__main__":
    list_user=[52853,31226,167471,8066 ]
    main(list_user)
    

import sys
sys.executable


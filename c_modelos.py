import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
from ipywidgets import interact ## para análisis interactivo

####Paquete para sistemas de recomendación surprise
###Puede generar problemas en instalación local de pyhton. Genera error instalando con pip

from surprise import Reader, Dataset
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD
from surprise.model_selection import train_test_split



#### conectar_base_de_Datos

conn=sql.connect('db_books2')
cur=conn.cursor()

#### ver tablas disponibles en base de datos ###

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
cur.fetchall()


#######################################################################
######## Sistema de recomendación basado en contenido - Manual ########
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


############################################################################
#####1. Sistema de recomendación filtro colaborativo basado en usuario #####
############################################################################


ratings=pd.read_sql('select * from ratings_final', conn)

###### leer datos desde tabla de pandas
reader = Reader(rating_scale=(0, 10))

###las columnas deben estar en orden estándar: user item rating
data   = Dataset.load_from_df(ratings[['user_id','isbn','book_rating']], reader)



models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline()] 
results = {}

for model in models:
 
    CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
    result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
             rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
    results[str(model).split("algorithms.")[1].split("object ")[0]] = result


performance_df = pd.DataFrame.from_dict(results).T
performance_df.sort_values(by='RMSE')


param_grid = { 'sim_options' : {'name': ['msd','cosine'], \
                                'min_support': [5], \
                                'user_based': [False, True]}
             }

gridsearchKNNWithMeans = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse'], \
                                      cv=2, n_jobs=2)
                                    
gridsearchKNNWithMeans.fit(data)

gridsearchKNNWithMeans.best_params["rmse"]
gridsearchKNNWithMeans.best_score["rmse"]


################# Realizar predicciones

trainset = data.build_full_trainset()


sim_options       = {'name':'msd','min_support':5,'user_based':True}
model = KNNWithMeans(sim_options=sim_options)
model=model.fit(trainset)


predset = trainset.build_anti_testset() 
predictions = model.test(predset) ### función muy pesada
predictions_df = pd.DataFrame(predictions)
prediction.shape

def recomendaciones(user_id,n_recomend=10):
    
    predictions_userID = predictions_df[predictions_df['uid'] == user_id].\
                    sort_values(by="est", ascending = False).head(n_recomend)

    recomendados = predictions_userID[['iid','est']]
    recomendados.to_sql('reco',conn,if_exists="replace")
    
    recomendados=pd.read_sql('''select a.*, b.book_title 
                             from reco a left join books_final b
                             on a.iid=b.isbn ''', conn)

    return(recomendados)

np.set_printoptions(threshold=sys.maxsize)
predictions_df['uid'].unique()[:20] 

us1=recomendaciones(user_id=179733,n_recomend=20)
us1




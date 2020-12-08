import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
from PIL import Image
import pickle
from pickle import dump

st.set_page_config(layout="wide")


filename = 'ModelSVM.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
ModelSVM, Tfidf_vect = loaded_model
#print(ModelSVM)
df = pd.read_csv('tweets_base_madre.csv', encoding ='utf-8')

dfFirst = df
dfFirst.drop('id', inplace=True, axis=1)
dfFirst.drop('user id', inplace=True, axis=1)
dfFirst.drop('tidy_text', inplace=True, axis=1)
dfFirst.drop('clasification', inplace=True, axis=1)

df = pd.read_csv('tweets_base_madre.csv', encoding ='utf-8')
dfClass = df
dfClass.drop('id', inplace=True, axis=1)
dfClass.drop('user id', inplace=True, axis=1)
dfClass.drop('tidy_text', inplace=True, axis=1)

st.write("""
# What happen if you mix Twitter, the president of México and natural language processing?
Nowdays, there is a lot of information in twitter and a bunch of people using this social media dialy (which creates more information) so… what happen if you grab a bunch of tweets that have words related to the politics of México and mainly, of course, Andrés Manuel López Obrador (actual president of México) and do an analysis? in this post we will show you the analysis that we did in the courses of natural language processing and pattern recognition in the university of Sonora.
""")

st.markdown('''
## Spoiler alert!
Our main goal was to train a model to tell us if a text was liberal or conservative, the evaluation criterion of our project is that our model yields a percentage of how much for or against a text is about the goverment (and more specific, about AMLO) given by a user.

### Test our final model by entering a text!
''')

#dfFinal = pd.read_csv('dfFinal.csv', encoding ='utf-8')
#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(dfFinal['TokenizeTweetsTidy_text'],dfFinal['clasification'],test_size=1)
#Tfidf_vect = TfidfVectorizer(max_features=50000)
#Tfidf_vect.fit(dfFinal['TokenizeTweetsTidy_text'])

input = st.text_input(label='Insert a text in spanish', value='Amlo presidente')

prediccionEjemplo = Tfidf_vect.transform([input])
#resultado2 = ModelSVM.predict(prediccionEjemplo)
resultadoPorcentaje = ModelSVM.predict_proba(prediccionEjemplo)

aux = []
for i in resultadoPorcentaje:
    for j in i:
        conservador = j
        aux = [j]

if conservador <= 0.5:
    st.write('Result: In favor of the goverment')
else:
    st.write('Result: Against the goverment')
st.write('Result: {}'.format(resultadoPorcentaje))
st.write("""
The f1 score will be used since it is used to combine the precision and recall measures into a single value.
This is practical because it makes it easier to compare the combined performance of accuracy and comprehensiveness between various solutions.

Our model would be considered successful if it managed to classify 100% when the model has no false positives and no false negatives, but we know that this is impossible and if it happened it would show that our model learned the data by memory and that is wrong. So what we consider a success is that we can reduce false positives and false negatives to the maximum.
""")
st.write("""
## Here is how we did it!!! 🤘😎
The firts thing we did was the information gathering, using the twitter API we collect as many tweets as we could (collecting 7985 tweets in total).

    df = pd.read_csv('../data/tweets_base_madre.csv', encoding ='utf-8')
""")
st.dataframe(dfFirst) 
st.write("""
Then, we classify them manually, creating a new column in our data frame named "classification". In that column we putted a number from 0 to 3, depending the content of the tweet.

0. If the content of the tweet was in favor of the goverment
1. If the content of the tweet was against the goverment
2. If the content of the tweet was neutral
3. If the content of the tweet wasn't related to the subject.
""")
st.dataframe(dfClass) 
st.write("""
### Before going deeper with tokenization, classification, etc. we did an analysis of which hashtags and arrobas people use when they mention AMLO and also the number of likes that the tweets have depending on their classification using the column "full_text".
## Hashtags:
""")
col1, col2 = st.beta_columns(2)

with col1:
    with open('Hashtags/Frecuencia de 260 HashTags de tweets liberales.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=480, scrolling=True)
    with open('Hashtags/Frecuencia de 436 HashTags de tweets conservadores.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=480, scrolling=True)
    
with col2:
    with open('Hashtags/Frecuencia de 415 HashTags de tweets neutrales.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=480, scrolling=True)
    with open('Hashtags/Frecuencia de 1110 HashTags de tweets Totales.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=480, scrolling=True) 
    

col1, col2, col3 = st.beta_columns(3)

with col1:
    with open('Hashtags/graficaHashtagsTop10.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=550, height=720, scrolling=True)

dataPlot = Image.open('Hashtags/AmloHashTags.png')
with col2:
    st.image(dataPlot, caption='WordCloud de Hashtags mas frecuentes', width=480, height=500)

with col3:
    with open('Hashtags/graficaHashtagsTopTotal.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=550, height=720, scrolling=True)

#st.write("""
### Arrobas.
#poner todo lo de arrobas
#
#""")

st.write("""
## Likes
""")
with open('graficaLikesTopTotal.html', 'r') as f:
    html_string = f.read()
components.html(html_string, width=800, height=480, scrolling=True)

st.write(
'''
## Now, let's go deeper...
After finishing the analysis of the hashtags, arrobas and likes, the next thing we did was create a new column with the name of "cleanTweets". In that column we applied some cleaning (with the help of regular expressions); the cleaning we did consisted of:

* Lemmatize.
* Transform all the text to lowercase.

#### Remove:
* Accents.
* Stop words.
* At symbols ( @ ).
* Hashtag symbols (#).
* Hypertexts.
* Suspension points.
* Isolated characters.
* Isolated numbers.
* Dates.
* Possible laughs (ja, jaja, ha, haha, je, jeje, etc).
* Certain symbols (symbols like {,}, [,], " , " , …)
~~~
def cleanTxt(text):
    # Cambia texto a minusculas
    text = text.lower()
    text = re.sub(r'@', '', text) #quita las @menciones
    text = re.sub(r'#', '', text) #quita los # simbolos
    text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+','',text) #quita los hyper textos
    # Expresiones regulares para remplazar simbolos por espacios 
    text = re.sub('[\ n'(){}\[\]\|,;\"\“\”\‘\’\'\«\»!¡?¿]', '', text)
    # Elimina Manejo de puntos suspensivos
    text = re.sub('\.[\.]+', '', text)
    # Elimina Manejo de caracteres aislados
    text = re.sub('\s.\s', '', text)
    # Elimina Manejo de números aislados
    text = re.sub('\s[0-9]+\s', '', text)
    # Elimina Manejo de fechas
    text = re.sub('\d\d/\d\d/\d\d|\d\d/\d\d/\d\d\d\d', '', text)
    # Se establece token de risa
    text = re.sub(r'\s([jJ][AaEeIi])+\s', r' <risa> ', text)
    text = re.sub(r'(ja|je|ji|JA|JE|JI|\s a|\s e|\s i)([jJ][AaEeIi])(\w?)', r' <risa> ', text)
    text = re.sub(r'\s((ha|Ha|he|HE)[hH][AaEeIi])+\w', r' <risa> ', text)
    return text

    df['CleanTweets'] = df['full_text'].apply(cleanTxt)
    df['CleanTweets'].dropna(inplace=True)
~~~
'''
)

df = pd.read_csv('dfFinal.csv', encoding ='utf-8')
dfClean = df
dfClean.drop('id', inplace=True, axis=1)
dfClean.drop('user id', inplace=True, axis=1)
dfClean.drop('tidy_text', inplace=True, axis=1)
dfClean.drop('TokenizeTweetsTidy_text', inplace=True, axis=1)


st.dataframe(dfClean) 

st.write(
'''
Note: Before taking the next step, we made sure that there were no repeated tweets (in case there was more than one of the same, we would keep the first one and delete the others)

The next step we did was to create another column named "TokenizeTweetsTidy_text", and that column contains the information of the column "cleanTweets" mentioned before BUT we applied tokenization to that column (agregar el porque creamos una columna?? es bueno tener el texto original).
~~~
#Tokenizar
df['TokenizeTweetsTidy_text'] = df.apply(lambda row: nltk.word_tokenize(row['CleanTweets']), axis=1)
#comillas a las palabras
df['TokenizeTweetsTidy_text'] = df.TokenizeTweetsTidy_text.astype(str)
~~~
'''
)

df = pd.read_csv('dfFinal.csv', encoding ='utf-8')
dfClean2 = df
dfClean2.drop('id', inplace=True, axis=1)
dfClean2.drop('user id', inplace=True, axis=1)
dfClean2.drop('tidy_text', inplace=True, axis=1)
st.dataframe(dfClean2) 

st.write(
'''
Next, we took (from the column "clasification") those tweets that have clasification=0 and clasification=1 and we created a new dataframe with those two together.
~~~
liberales = df[df["clasification"]==0]
conservadores  = df[df["clasification"]==1]

libCon = pd.concat([liberales,conservadores]).reset_index()
~~~
👀 In the dataFrame libCon are 1639 Tweets of the class 0 (Liberales) and 3164 Tweets of the class 1 (Conservadores), giving us a total of 4803 Tweets.
'''
)

df = pd.read_csv('libCon.csv', encoding ='utf-8')
libCon = df
libCon.drop('id', inplace=True, axis=1)
libCon.drop('user id', inplace=True, axis=1)
libCon.drop('tidy_text', inplace=True, axis=1)
st.dataframe(libCon)

st.write(
    '''
## Then, we vectorize the tweets that are in the column "TokenizeTweetsTidy_text" with Tf-idf.
To see the tf-idf data set we use TSNEVisualizer, this one creates an inner transformer pipeline that applies such a decomposition first (SVD with 50 components by default), then performs the t-SNE embedding. The visualizer then plots the scatter plot, coloring by cluster or by class, or neither if a structural analysis is required.
    '''
)

col1, col2 = st.beta_columns(2)
with col1:
    st.write(
    '''
    ~~~
    def GraficaData(df,nombreCol):
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            # execute code that will generate warnings
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df[nombreCol],df['clasification'],test_size=1)
            y = Train_Y
            Encoder = LabelEncoder()
            Train_Y = Encoder.fit_transform(Train_Y)
            Test_Y = Encoder.fit_transform(Test_Y)

            Tfidf_vect = TfidfVectorizer(max_features=50000)
            Tfidf_vect.fit(df[nombreCol])

            Train_X_Tfidf = Tfidf_vect.transform(Train_X)
            Test_X_Tfidf = Tfidf_vect.transform(Test_X)
            #print(Tfidf_vect.vocabulary_)
            len(Tfidf_vect.vocabulary_)
            #print(Train_X_Tfidf)
            warnings.filterwarnings("ignore")
            tsne = TSNEVisualizer(cmap='PuOr')
            warnings.filterwarnings("ignore")
            tsne.fit(Train_X_Tfidf, y)
            warnings.filterwarnings("ignore")
            tsne.show()

    GraficaData(libCon,'TokenizeTweetsTidy_text')
    ~~~
    '''
    )
    
with col2:
    dataPlot = Image.open('dataPlotAi.png')
    st.image(dataPlot, caption='Tf-idf data', width=720, height=480)

st.write(
'''
## Before going into the clasification… what happen if we want to know (passing specific percentage of dataset) the degree of success achieved of our chosen model?

### We can know that by doing the learning curve!
'''
)

col1, col2 = st.beta_columns(2)

with col1:

    st.write(
    '''
    ~~~
    # Ploting Learning Curve
    # Creating CV training and test scores for various training set sizes
    X, _, y, _, Tfidf_vect = separaData(df,'TokenizeTweetsTidy_text',1)
    cv = StratifiedKFold(n_splits=10)
    estimator = LogisticRegression(solver='lbfgs',max_iter=100 , C=1)
    train_sizes, train_scores, test_scores = learning_curve(estimator,X, y, cv=cv, scoring='f1_weighted', n_jobs=-1,train_sizes=np.linspace(0.3, 1.0, 10))

    # Creating means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Creating means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()

    layout = go.Layout(title='Learning Curve Logistic Regression',
                    xaxis=dict(title='Training Set Size'),
                    yaxis=dict(title='Accuracy Score'))

    fig = go.Figure(layout=layout)


    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        line_color='rgb(230,171,2)',
        line = dict(width=4, dash='dash'),
        name='Training score',
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        line_color='rgb(117,112,179)',
        line = dict(width=4),
        mode='lines+markers',
        name='Cross-validation score',
        
    ))

    fig.update_traces(mode='lines')
    fig.write_html("LearningCurveSVM.html")
    fig.show()
    ~~~
    '''
    )

    with open('LearningCurveSVM.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=480, scrolling=True)

with col2:
    st.write(
    '''
    ~~~
    # Ploting Learning Curve
    # Creating CV training and test scores for various training set sizes
    X, _, y, _, Tfidf_vect = separaData(df,'TokenizeTweetsTidy_text',1)
    cv = StratifiedKFold(n_splits=10)
    estimator = SVC(gamma=1, C=1,probability=True, kernel= 'rbf')
    train_sizes, train_scores, test_scores = learning_curve(estimator,X, y, cv=cv, scoring='f1_weighted', n_jobs=-1,train_sizes=np.linspace(0.3, 1.0, 10))

    # Creating means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Creating means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig = go.Figure()

    layout = go.Layout(title='Learning Curve SVM',
                    xaxis=dict(title='Training Set Size'),
                    yaxis=dict(title='Accuracy Score'))

    fig = go.Figure(layout=layout)


    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        line_color='rgb(230,171,2)',
        line = dict(width=4, dash='dash'),
        name='Training score',
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes, y=test_mean,
        line_color='rgb(117,112,179)',
        line = dict(width=4),
        mode='lines+markers',
        name='Cross-validation score',
        
    ))

    fig.update_traces(mode='lines')
    fig.write_html("LearningCurveRegLog.html")
    fig.show()
    ~~~
    '''
    )


    with open('LearningCurveRegLog.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=800, height=480, scrolling=True)

st.write("""
*According to the learning curves of the two models, the two graphs tell us the same thing:
more training data is needed to be able to have a smaller error.*
""")

st.write(
'''
## ¡Here starts the clasification!
To classify we used logistic regression and SVM to see which one classifies the best.
~~~
def ModeloClasificadorKFoldCV(modeloToTrain, df,nombre):
    X, _, y, _, Tfidf_vect = separaData(df,'TokenizeTweetsTidy_text',1)

    if modeloToTrain == LogisticRegression:
        myModelo = modeloToTrain(solver='lbfgs',max_iter=100 , C=1)
    elif modeloToTrain == SVC:
        myModelo = modeloToTrain(gamma=1, C=1,probability=True, kernel= 'rbf')

    cv = KFold(n_splits=10, random_state=None, shuffle=False)
    scores = []

        for train_index, test_index in cv.split(X):
            Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y = X[train_index], X[test_index], y[train_index], y[test_index]
            myModelo.fit(Train_X_Tfidf, Train_Y)
            scores.append(myModelo.score(Test_X_Tfidf, Test_Y))
    ~~~
    ### Logistic regression.
    Why did we choose the logistic regression model? because it is the easiest model to train and also gives a good (and fast) classification.

    The first thing we did was define our k-fold and after that we trained!

    Then we got the predictions, the confusion matrix and the Roc curve of the model.
    ~~~
    modelLogisticRegression = ModeloClasificadorKFoldCV(LogisticRegression, libCon, "LogisticRegression")
    ~~~
'''
)

col1, col2, col3 = st.beta_columns(3)

with col1:
    dataPlot = Image.open('logRegReportAi.png')
    st.image(dataPlot, caption='', width=620, height=380)
with col2:
    with open('matrizLogisticRegression.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=600, height=500, scrolling=True)
with col3:
    with open('rocaucLogisticRegression.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=650, height=500, scrolling=True)
    
st.write(
'''

As can be seen in the graphs, we can conclude that the logistic regression classifies conservative tweets very well, but liberal tweets very poorly, this is observed from the confusion matrix and the f1 score.
From the roc curve, we can say that our classes, especially, the classification of the "liberal" class is very bad. For this we will try to solve our problems using another model which would be the SVM, since we want to improve the f1 score of the "liberal" class.

'''
)
st.write(
'''
### SVM (Support-Vector Machine)
Why did we choose the SVM model? because is a fast and dependable classification algorithm that performs very well with a limited amount of data to analyze. 

The first thing we did was define our k-fold and after that we trained!

Then we got the predictions, the confusion matrix and the Roc curve.
~~~
modelSVC = ModeloClasificadorKFoldCV(SVC, libCon, "SVC")
~~~
'''
)
col1, col2, col3 = st.beta_columns(3)
with col1:
    dataPlot = Image.open('svmReportAi.png')
    st.image(dataPlot, caption='', width=620, height=380)
with col2:
    with open('matrizSVC.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=600, height=500, scrolling=True)
with col3:
    with open('rocaucSVC.html', 'r') as f:
        html_string = f.read()
    components.html(html_string, width=650, height=500, scrolling=True)
    
st.write(
'''
The results of the SVM do not differ much from the logistic regression model, the good thing here is that it increased what we wanted; the f1 score of the "liberal" class.

So after looking at the SVM and logistic regression results, we will use all the available data to classify a final model using SVM:

~~~
def ModeloClasificadorFinal(modeloToTrain, df,nombre):
    Train_X_Tfidf, Test_X_Tfidf, Train_Y, Test_Y, Tfidf_vect = separaData(df,'TokenizeTweetsTidy_text',1)
    
    if modeloToTrain == LogisticRegression:
        myModelo = modeloToTrain(solver='lbfgs',max_iter=100, C=1)
    elif modeloToTrain == SVC:
            myModelo = modeloToTrain(gamma=1, C=1,probability=True, kernel= 'rbf') 
        
    myModelo.fit(Train_X_Tfidf,Train_Y)
    
    return myModelo

modelSVCFinal = ModeloClasificadorFinal(SVC, libCon, "Maquina de vector de soportes")
~~~
'''
)
st.write(
'''

## Conclusion
After seeing the results of the two models, we can conclude that to improve our model we must have more "liberal" tweets, since it can be seen in the two models that due to this problem our f1 score of the "liberal" class was extremely decreased and this makes our classifiers not 100% reliable.
'''
)



#components.html(html_string,height=600)

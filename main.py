import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''## Простое приложение для предсказания цветов ириса
Это приложение предсказывает тип цветка ириса!''')

st.sidebar.header('Параметры пользовательского ввода')

def user_input_features():
    sepal_length = st.sidebar.slider('Длина чашелистика', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Ширина чашелистика', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Длина депестка', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Ширина лепестка', 0.1, 2.5, 0.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Параметры пользовательского ввода')

st.write(df)

iris = datasets.load_iris()

X = iris.data
Y = iris.target

clf = RandomForestClassifier()

clf.fit(X, Y)

prediction = clf.predict(df)

prediction_proba = clf.predict_proba(df)



st.subheader('Метки классов и соответствующие им порядковые номера')
st.write(iris.target_names)

st.subheader('Прогноз')
st.write(iris.target_names[prediction])

st.subheader('Вероятность прогноза')
st.write(prediction_proba)

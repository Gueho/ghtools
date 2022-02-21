def preprocess():
    '''preprocess cat/num features'''
    import numpy as np
    import pandas as pd

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_selector as selector

    num_transformer = Pipeline([('imputer', SimpleImputer()),
                                ('scaler', StandardScaler())])
    cat_transformer = OneHotEncoder(handle_unknown = 'ignore')

    preprocessor = ColumnTransformer([
        ("num_tr", num_transformer, selector(dtype_exclude = object)),
        ("cat_tr", cat_transformer, selector(dtype_include = object))],
    remainder = 'passthrough')

    return preprocessor

def make_pipe(type = 'Regression'):
    '''make a whole pipeline with Regression/Classification'''
    preproc = preprocess()

    from sklearn.linear_model import SGDRegressor, SGDClassifier
    from sklearn.pipeline import Pipeline

    if type == 'Regression':
        pipe = Pipeline([
            ('preprocess', preproc),
        ('Regress', SGDRegressor())])

    if type == 'Classification':
        pipe = Pipeline([
            ('preprocess', preproc),
        ('Classify', SGDClassifier())])

    return pipe

def predict(X_train, y_train, X_test, type = 'Regression'):
    pipe = make_pipe(type = 'Regression')

    pipe.fit(X_train, y_train)
    prediction = pipe.predict(X_test)
    return prediction

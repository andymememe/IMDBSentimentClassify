from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.model_selection import RandomizedSearchCV as CV
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from joblib import dump

import os


def main():
    sw = list(stopwords.words("english"))
    data_src = "D:\\Reference\\aclImdb"
    tags = ['neg', 'pos']

    gbm_param_grid = {
        'n_estimators': range(5,20),
        'max_depth': range(6,20),
        'learning_rate': [.4, .45, .5, .55, .6],
        'colsample_bytree': [.6, .7, .8, .9, 1],
        'min_child_weight':range(1,6,2)
    }

    # Training
    x_train = []
    y_train = []
    for tag in tags:
        for aFile in os.listdir(f"{data_src}\\train\\{tag}"):
            with open(f"{data_src}\\train\\{tag}\\{aFile}", "r", encoding="utf-8") as f:
                x_train.append(f.read().strip())
                y_train.append(tags.index(tag))

    tfidf = TFIDF(stop_words=sw)
    x_train_tfidf = tfidf.fit_transform(x_train)

    xgb = XGBClassifier()
    xgb_random = CV(
        param_distributions=gbm_param_grid,
        estimator=xgb,
        scoring="accuracy",
        verbose=1,
        n_iter=50, 
        cv=5,
        n_jobs=-1
    )
    xgb_random.fit(x_train_tfidf, y_train)

    print("Search log: ", xgb_random.cv_results_)
    print("Best parameters found: ", xgb_random.best_params_)
    print("Best accuracy found: ", xgb_random.best_score_)


    # Testing
    x_test = []
    y_test = []
    for tag in tags:
        for aFile in os.listdir(f"{data_src}\\test\\{tag}"):
            with open(f"{data_src}\\test\\{tag}\\{aFile}", "r", encoding="utf-8") as f:
                x_test.append(f.read().strip())
                y_test.append(tags.index(tag))
    x_test_tfidf = tfidf.transform(x_test)
    y_pred = xgb_random.predict(x_test_tfidf)

    print("Acc:", accuracy_score(y_test, y_pred))
    print("Rec:", recall_score(y_test, y_pred))
    print("Pre:", precision_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))

    # Save The Model
    dump(tfidf, "model/tfidf.pkl")
    dump(xgb_random, "model/xgb.pkl")

if __name__ == "__main__":
    main()
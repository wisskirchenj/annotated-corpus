from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk import SnowballStemmer, download
from nltk.stem import WordNetLemmatizer

download('wordnet')
lemmatizer = WordNetLemmatizer()
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=182)
classifier = RandomForestClassifier(n_estimators=15, max_features=2, oob_score=True, random_state=127)
classifier.fit(X_train, y_train)
print(f'{classifier.score(X_test, y_test):.3f}')
print(SnowballStemmer('english').stem('generously'))
print(SnowballStemmer('english').stem('dangerously'))
print(lemmatizer.lemmatize('generously'))
print(lemmatizer.lemmatize('supercars', pos='n'))
print(lemmatizer.lemmatize('cars', pos='n'))
print(lemmatizer.lemmatize('men', pos='n'))


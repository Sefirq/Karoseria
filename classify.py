from pandas import read_csv, DataFrame
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix
import pydotplus


train = read_csv('result.csv')
test = read_csv('result.csv')
training_labels = list(train.loc[:, 'class'])
testing_labels = list(test.loc[:, 'class'])
del train['class']
del test['class']
del train['HpW']
del test['HpW']
print(train.columns.values)
training_features = train.values.tolist()
testing_features = test.values.tolist()
classifier = DecisionTreeClassifier()
classifier.fit(training_features, training_labels)
predicted_labels = classifier.predict(testing_features)
print("Ważność poszczególnych cech:")
print(classifier.feature_importances_)
for i, j in zip(testing_labels, predicted_labels):
    print(i, j)
print("   Predicted")
print("   d s v")
lol = ['d', 's', 'v']
x = confusion_matrix(testing_labels, predicted_labels)
for i, row in enumerate(x):
    print(lol[i], row)
dot_data = export_graphviz(classifier, feature_names=train.columns.values,class_names=['d','s','v'], out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("cars.pdf")

from sklearn.svm import LinearSVC, SVC
import pickle
from sklearn.externals import joblib
'''
with open('data.p', mode='rb') as f:
    data = pickle.load(f)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
'''

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
svc.fit(X_train, y_train)
# Check the score of the SVC
print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
#print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
joblib.dump(svc, 'svc.pkl')

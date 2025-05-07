 # Precision, Recall, F1, IoU, mAP
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(
    y_test.argmax(axis=1),
    y_pred.argmax(axis=1),
    target_names=classes
))  
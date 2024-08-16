import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
import joblib

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_training_data', type=str, help='Input CSV file path to training')
    parser.add_argument('--input_test_data', type=str, help='Input CSV file path to test')
    parser.add_argument('--target_column', type=str, help='Name of the target column')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of boosting stages to be run')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum depth of the tree')
    parser.add_argument('--max_features', type=int, default="log2")
    parser.add_argument('--min_samples_split', type=float, default="2")
    parser.add_argument('--min_samples_leaf', type=float, default="1")
    parser.add_argument('--loss', type=float, default="1")
    parser.add_argument('--criterion', type=float, default="friedman_mse")
    parser.add_argument('--subsample', type=float, default="1.0")
    parser.add_argument('--output_model', type=str, help='Path to save the trained model', default = "./model.pkl")
    parser.add_argument('--output_metrics', type=str, help='Path to save the model performance metrics')
    parser.add_argument('--register_model', type=bool, default = True)

    args = parser.parse_args()

    return args

args = argument_parser()

# Load data
train_data = pd.read_csv(args.input_training_data)
test_data = pd.read_csv(args.input_test_data)
X_train = train_data.drop(columns=[args.target_column])
y_train = train_data[args.target_column]
X_test = test_data.drop(columns=[args.target_column])
y_test = test_data[args.target_column]

mlflow.log_param("N estimators", args.n_estimators)
mlflow.log_param("Learning rate", args.learning_rate)
mlflow.log_param("Max Depth", args.max_depth)
mlflow.log_param("Max Features", args.max_features)
mlflow.log_param("Min Samples Split", args.min_samples_split)
mlflow.log_param("Min Samples Leaf", args.min_samples_leaf)
mlflow.log_param("Loss Function", args.loss)
mlflow.log_param("Criterion", args.criterion)
mlflow.log_params("Subsample", args.subsample)

# Train Gradient Boosting Classifier
model = GradientBoostingClassifier(
    n_estimators=args.n_estimators,
    learning_rate=args.learning_rate,
    max_depth=args.max_depth,
    max_features=args.max_features,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    loss=args.loss,
    criterion=args.criterion,
    subsample=args.subsample
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("Accuracy", accuracy)

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
mlflow.log_metric("AUC", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
fig = plt.figure(figsize=(6, 4))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig("ROC-Curve.png")
mlflow.log_artifact("ROC-Curve.png")

joblib.dump(model, args.output_model)

model_output = args.output_model

if args.register_model:

    signature = infer_signature(train_data, model.predict(y_train))
    mlflow.sklearn.log_model(model, "job_classifier", signature=signature)
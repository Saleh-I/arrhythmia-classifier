import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
# Load CSV file
df = pd.read_csv("../experiments/exp1/output/evaluation.csv")  # Replace with your actual file path

# Extract ground truth and predictions
y_true = df["_gt"]
y_pred = df["_pred"]

print(confusion_matrix(y_true, y_pred))
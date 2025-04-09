# Fix for tqdm notebook issues in VS Code
import tqdm
tqdm.tqdm.__init__ = tqdm.std.tqdm.__init__
tqdm.tqdm.__del__ = lambda self: None

# Import libraries
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv(r"C:\Users\ABIRAINA\OneDrive\Desktop\guvi_project\miniproject3\Employee-Attrition - Employee-Attrition.csv")

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target

# Features and target
X = data[['OverTime','JobInvolvement','YearsAtCompany']]
y = data["Attrition"]

# Scale features (important for some models like SVC)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split before SMOTE (only apply SMOTE to training data!)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Run LazyClassifier on balanced data
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_bal, X_test, y_train_bal, y_test)

# Show results
print("\nModel Benchmarking Results (Balanced Dataset):\n")
print(models)

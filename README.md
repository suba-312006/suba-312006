import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("US_Accidents_Dec21_updated.csv")  # Replace with your actual path

# Selecting features for severity prediction
df = df[['Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)', 
         'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']]

# Drop missing values
df.dropna(inplace=True)

# Convert multi-class severity to binary classification (e.g., Low: 1-2, High: 3-4)
df['Severity'] = df['Severity'].apply(lambda x: 0 if x <= 2 else 1)

# Define features and label
X = df.drop('Severity', axis=1)
y = df['Severity']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline with scaler and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting correlation heatmap for analysis
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()- 👋 Hi, I’m @suba-312006
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
suba-312006/suba-312006 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

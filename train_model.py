import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load and prepare the dataset
df = pd.read_csv('framingham.csv')
df.drop(['education'], axis=1, inplace=True)
df.rename(columns={'male': 'Sex_male'}, inplace=True)
df.dropna(inplace=True)

# Define features and target
X = df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']]
y = df['TenYearCHD']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=4)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Model and scaler saved successfully!")

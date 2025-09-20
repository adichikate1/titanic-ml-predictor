from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    models = None
    tables = None

    if request.method == "POST":
        # Example Titanic-like training data
        train_data = {
            "PassengerId": range(1, 11),
            "Survived": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
            "Pclass": [3, 1, 3, 1, 3, 3, 2, 3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina", "Futrelle, Mrs. Jacques Heath",
                "Allen, Mr. William Henry", "Moran, Mr. James",
                "Myles, Mr. Thomas Francis", "McCarthy, Mr. Timothy J",
                "Palsson, Master. Gosta Leonard", "Johnson, Mr. William Cahoone"
            ],
            "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "male", "male"],
            "Age": [22, 38, 26, 35, 35, np.nan, 30, 54, 2, 47],
            "SibSp": [1, 1, 0, 1, 0, 0, 0, 0, 3, 0],
            "Parch": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450", "330877",
                       "A/5 21172", "17463", "349909", "347742"],
            "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 7.75, 51.8625, 21.075, 7.25],
            "Cabin": [np.nan, "C85", np.nan, "C123", np.nan, np.nan, np.nan, "E46", np.nan, np.nan],
            "Embarked": ["S", "C", "S", "S", "S", "Q", "Q", "S", "S", "S"]
        }

        train_df = pd.DataFrame(train_data)

        # Handle uploaded file
        file = request.files["file"]
        if file.filename.endswith(".csv"):
            file_data = pd.read_csv(file)
        elif file.filename.endswith((".xls", ".xlsx")):
            file_data = pd.read_excel(file)
        else:
            return "Unsupported file format. Please upload CSV or Excel."

        # Fill missing values in uploaded file
        file_data["Age"] = file_data["Age"].fillna(file_data["Age"].median())
        file_data["Fare"] = file_data["Fare"].fillna(file_data["Fare"].mode()[0])

        # --------------------------
        # Train preprocessing
        # --------------------------
        X = train_df.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
        y = train_df["Survived"]

        # Encode categorical variables
        X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Train-test split
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train models
        log_reg = LogisticRegression(max_iter=200).fit(x_train, y_train)
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train, y_train)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(x_train, y_train)

        # Accuracies
        models = {
            "Logistic Regression": round(log_reg.score(x_val, y_val), 3),
            "Random Forest": round(rf.score(x_val, y_val), 3),
            "Gradient Boosting": round(gb.score(x_val, y_val), 3),
        }

        # --------------------------
        # Preprocess uploaded file (file_data → file_x)
        # --------------------------
        file_x = file_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
        file_x = pd.get_dummies(file_x, columns=["Sex", "Embarked"], drop_first=True)

        # Align columns with training features
        file_x = file_x.reindex(columns=X.columns, fill_value=0)

        # Impute missing
        file_x = pd.DataFrame(imputer.transform(file_x), columns=X.columns)

        # Final prediction
        final_model = rf.fit(X, y)
        predictions = final_model.predict(file_x)

        # Add predictions to uploaded file
        file_data["Survived_Prediction"] = predictions

        # Convert DataFrame to HTML table
        tables = [file_data.head(20).to_html(classes="data table table-bordered", index=False)]

    return render_template("index.html", models=models, tables=tables)

if __name__ == "__main__":
    app.run(debug=True)



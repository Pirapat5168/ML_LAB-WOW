from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, pca_components=150, use_pca=True):
    # เปลี่ยนเป็น list ว่าง ถ้าไม่ใช้ PCA -> Pipeline จะมีแค่ scaler ขั้นตอนเดียว
    steps = [("scaler", StandardScaler())]

    if use_pca:
        steps.append(
            ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                        whiten=True, random_state=42))
        )

    scaler = Pipeline(steps)
    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Create SVM model
    model = SVC(
        kernel="rbf", C=10, gamma="scale", cache_size=1000
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):

    # Apply the same scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions
def train_svm_gridsearch(X_train, y_train, pca_components=150, use_pca=True):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(
            ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                        whiten=True, random_state=42))
        )
    scaler = Pipeline(steps)

    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", 0.01],
    }

    search = GridSearchCV(
        SVC(cache_size=1000),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
    )
    search.fit(X_train_scaled, y_train)

    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_ * 100:.2f}%")

    return search.best_estimator_, scaler
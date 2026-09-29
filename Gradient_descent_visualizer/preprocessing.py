import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def load_data():

    # Load dataset
    df = pd.read_csv("data/insurance.csv")

    return df


def preprocess_data():


    # Load dataset
    df = load_data()

    
    # Separate features and target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    
    # One-hot encode categorical data
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=float
    )

    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    
    # Standard scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    
    # Convert target to NumPy
    y_train = y_train.to_numpy()

    y_test = y_test.to_numpy()

    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler)


def create_polynomial_features(
    X_train_scaled,
    X_test_scaled,
    degree=2
):

    
    # Create polynomial features
    
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    
    X_train_poly = poly.fit_transform(
        X_train_scaled
    )

    
    X_test_poly = poly.transform(
        X_test_scaled
    )

    
    # Scale polynomial features
    
    poly_scaler = StandardScaler()

    X_train_poly_scaled = poly_scaler.fit_transform(
        X_train_poly
    )

    X_test_poly_scaled = poly_scaler.transform(
        X_test_poly
    )

    return (
        X_train_poly_scaled,
        X_test_poly_scaled,
        poly,
        poly_scaler
    )
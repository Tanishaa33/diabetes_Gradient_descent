import numpy as np

# LINEAR REGRESSION USING GRADIENT DESCENT
class LinearRegressionGD:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=10000,
        tolerance=1.0,
        patience=100
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.patience = patience

        # Model parameters
        self.weights = None
        self.bias = 0.0

        # Training history
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []
        self.gradient_history = []

        # Actual epochs used
        self.n_epochs_used = 0


    def predict(self, X):

        return X @ self.weights + self.bias


    def calculate_loss(self, X, y):

        predictions = self.predict(X)

        error = predictions - y

        return np.mean(error ** 2)


    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize weights and bias

        self.weights = np.zeros(n_features)

        self.bias = 0.0

        # Reset history
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []
        self.gradient_history = []

        # Early stopping variables

        best_loss = float("inf")

        best_weights = self.weights.copy()

        best_bias = self.bias

        patience_counter = 0

        # Gradient Descent

        for epoch in range(self.epochs):

            # Forward propagation
            predictions = X @ self.weights + self.bias

            # Error
            error = predictions - y

            # Calculate gradients

            dw = (
                (2 / n_samples)
                * (X.T @ error)
            )

            db = (
                (2 / n_samples)
                * np.sum(error)
            )

            # Update parameters

            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

            # Calculate new loss

            new_predictions = (
                X @ self.weights + self.bias
            )

            new_error = (
                new_predictions - y
            )

            new_loss = np.mean(
                new_error ** 2
            )

            # Store history
            self.loss_history.append(
                float(new_loss)
            )

            self.weights_history.append(
                self.weights.copy()
            )

            self.bias_history.append(
                float(self.bias)
            )

            self.gradient_history.append(
                dw.copy()
            )

            # Early stopping
            improvement = (
                best_loss - new_loss
            )

            if improvement > self.tolerance:

                best_loss = new_loss

                best_weights = (
                    self.weights.copy()
                )

                best_bias = self.bias

                patience_counter = 0

            else:

                patience_counter += 1

            # Progress display

            if (
                epoch == 0
                or (epoch + 1) % 500 == 0
            ):

                print(
                    f"Epoch {epoch + 1:5d} "
                    f"| Loss: {new_loss:.4f} "
                    f"| Patience: "
                    f"{patience_counter}/"
                    f"{self.patience}"
                )

            # Stop training

            if patience_counter >= self.patience:

                print(
                    f"\nEarly stopping at "
                    f"epoch {epoch + 1}"
                )

                break

        # Restore best parameters

        self.weights = best_weights

        self.bias = best_bias

        self.n_epochs_used = (
            len(self.loss_history)
        )

        print(
            f"Epochs actually used: "
            f"{self.n_epochs_used}"
        )

        print(
            f"Best training loss: "
            f"{best_loss:.4f}"
        )

        return self


# POLYNOMIAL REGRESSION USING GRADIENT DESCENT

class PolynomialRegressorGD:

    def __init__(
        self,
        learning_rate=0.01,
        epochs=10000,
        tolerance=1.0,
        patience=100
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Early stopping
        self.tolerance = tolerance
        self.patience = patience

        # Model parameters
        self.weights = None
        self.bias = 0.0

        # Training history
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []
        self.gradient_history = []

        # Actual epochs used
        self.n_epochs_used = 0


    def predict(self, X):

        return X @ self.weights + self.bias


    def calculate_loss(self, X, y):

        predictions = self.predict(X)

        error = predictions - y

        return np.mean(error ** 2)


    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)

        self.bias = 0.0

        # Reset history
        self.loss_history = []
        self.weights_history = []
        self.bias_history = []
        self.gradient_history = []

        # Early stopping
        best_loss = float("inf")

        best_weights = self.weights.copy()

        best_bias = self.bias

        patience_counter = 0

        # Gradient Descent loop
        for epoch in range(self.epochs):

            # Forward pass
            predictions = (
                X @ self.weights
                + self.bias
            )

            # Error
            error = predictions - y

            # Gradients
            dw = (
                (2 / n_samples)
                * (X.T @ error)
            )

            db = (
                (2 / n_samples)
                * np.sum(error)
            )

            # Parameter update

            self.weights -= (
                self.learning_rate * dw
            )

            self.bias -= (
                self.learning_rate * db
            )

            # Loss AFTER update
            new_predictions = (
                X @ self.weights
                + self.bias
            )

            new_error = (
                new_predictions - y
            )

            new_loss = np.mean(
                new_error ** 2
            )

            # Store loss
            self.loss_history.append(
                float(new_loss)
            )

            # Store parameter history
            self.weights_history.append(
                self.weights.copy()
            )

            self.bias_history.append(
                float(self.bias)
            )

            self.gradient_history.append(
                dw.copy()
            )

            
            # Early stopping

            improvement = (
                best_loss - new_loss
            )

            if improvement > self.tolerance:

                best_loss = new_loss

                best_weights = (
                    self.weights.copy()
                )

                best_bias = self.bias

                patience_counter = 0

            else:

                patience_counter += 1

            
            # Progress display
        
            if (
                epoch == 0
                or (epoch + 1) % 500 == 0
            ):

                print(
                    f"Epoch {epoch + 1:5d} "
                    f"| Loss: {new_loss:.4f} "
                    f"| Patience: "
                    f"{patience_counter}/"
                    f"{self.patience}"
                )

            
            # Stop training
            
            if patience_counter >= self.patience:

                print(
                    f"\nEarly stopping at "
                    f"epoch {epoch + 1}"
                )

                break

        
        # Restore best parameters
      

        self.weights = best_weights

        self.bias = best_bias

        self.n_epochs_used = (
            len(self.loss_history)
        )

        print(
            f"Epochs actually used: "
            f"{self.n_epochs_used}"
        )

        print(
            f"Best training loss: "
            f"{best_loss:.4f}"
        )

        return self
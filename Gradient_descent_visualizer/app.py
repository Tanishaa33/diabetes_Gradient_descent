import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

from preprocessing import (
    preprocess_data,
    create_polynomial_features
)

from model import (
    LinearRegressionGD,
    PolynomialRegressorGD
)


# =========================================================
# LOAD DATA
# =========================================================

(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = preprocess_data()


# =========================================================
# TRAINING FUNCTION
# =========================================================

def train_model(model_type, learning_rate, epochs, degree):

    # ---------------------------------------------
    # LINEAR MODEL
    # ---------------------------------------------

    if model_type == "Linear Regression":

        model = LinearRegressionGD(
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            tolerance=1.0,
            patience=100
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions = model.predict(
            X_test_scaled
        )

    # ---------------------------------------------
    # POLYNOMIAL MODEL
    # ---------------------------------------------

    else:

        (
            X_train_poly_scaled,
            X_test_poly_scaled,
            poly,
            poly_scaler
        ) = create_polynomial_features(
            X_train_scaled,
            X_test_scaled,
            degree=int(degree)
        )

        model = PolynomialRegressorGD(
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            tolerance=1.0,
            patience=100
        )

        model.fit(
            X_train_poly_scaled,
            y_train
        )

        predictions = model.predict(
            X_test_poly_scaled
        )


    # =====================================================
    # METRICS
    # =====================================================

    mae = np.mean(
        np.abs(y_test - predictions)
    )

    mse = np.mean(
        (y_test - predictions) ** 2
    )

    rmse = np.sqrt(mse)

    r2 = (
        1 -
        np.sum((y_test - predictions) ** 2)
        /
        np.sum((y_test - np.mean(y_test)) ** 2)
    )


    # =====================================================
    # LOSS GRAPH
    # =====================================================

    fig, ax = plt.subplots()

    ax.plot(
        model.loss_history
    )

    ax.set_title(
        "Gradient Descent - Loss vs Epoch"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss (MSE)"
    )

    ax.grid(
        True
    )

    plt.tight_layout()


    # =====================================================
    # RESULTS
    # =====================================================

    results = f"""
Model: {model_type}

Learning Rate: {learning_rate}

Epochs Requested: {epochs}

Epochs Used: {model.n_epochs_used}

MAE: {mae:.4f}

RMSE: {rmse:.4f}

R²: {r2:.6f}
"""


    return results, fig


# =========================================================
# GRADIO INTERFACE
# =========================================================

with gr.Blocks(
    title="Gradient Descent Visualizer"
) as demo:

    gr.Markdown(
        """
        # 📈 Gradient Descent Visualizer

        Train a custom Gradient Descent regression model
        and visualize how the loss changes during training.
        """
    )


    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    with gr.Row():

        model_type = gr.Dropdown(
            choices=[
                "Linear Regression",
                "Polynomial Regression"
            ],
            value="Linear Regression",
            label="Model"
        )

        learning_rate = gr.Number(
            value=0.01,
            label="Learning Rate"
        )

        epochs = gr.Number(
            value=10000,
            label="Epochs"
        )

        degree = gr.Number(
            value=2,
            label="Polynomial Degree"
        )


    train_button = gr.Button(
        "🚀 Train Model"
    )


    # -----------------------------------------------------
    # OUTPUTS
    # -----------------------------------------------------

    results = gr.Textbox(
        label="Model Results",
        lines=10
    )

    loss_plot = gr.Plot(
        label="Gradient Descent Visualization"
    )


    # -----------------------------------------------------
    # BUTTON ACTION
    # -----------------------------------------------------

    train_button.click(
        fn=train_model,
        inputs=[
            model_type,
            learning_rate,
            epochs,
            degree
        ],
        outputs=[
            results,
            loss_plot
        ]
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.launch()
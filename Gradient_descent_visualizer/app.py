import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import (
    preprocess_data,
    create_polynomial_features
)

from model import (
    LinearRegressionGD,
    PolynomialRegressorGD
)


# LOAD DATA

(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler
) = preprocess_data()


# COLORS

BLACK = "#181818"
DARK = "#111111"
WHITE = "#FFFFFF"
GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
ACCENT = "#C8B6A6"

# Different colors for Parameter Learning graph
PARAMETER_COLORS = [
    "#8B7BB5",   # Purple
    "#6F9EC4",   # Blue
    "#72A98A",   # Green
    "#D99A62",   # Orange
    "#C97891",   # Pink
    "#5FA8A0"    # Teal
]


# TRAINING FUNCTION

def train_model(
    model_type,
    learning_rate,
    epochs,
    degree
):

    learning_rate = float(learning_rate)
    epochs = int(epochs)
    degree = int(degree)

    # LINEAR REGRESSION

    if model_type == "Linear Regression":

        model = LinearRegressionGD(
            learning_rate=learning_rate,
            epochs=epochs,
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

        X_used = X_train_scaled

        model_name = "Linear Regression"

        polynomial_features = "Not Used"

    # POLYNOMIAL REGRESSION

    else:

        (
            X_train_poly_scaled,
            X_test_poly_scaled,
            poly,
            poly_scaler
        ) = create_polynomial_features(
            X_train_scaled,
            X_test_scaled,
            degree=degree
        )

        model = PolynomialRegressorGD(
            learning_rate=learning_rate,
            epochs=epochs,
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

        X_used = X_train_poly_scaled

        model_name = "Polynomial Regression"

        polynomial_features = f"Degree {degree}"

    # METRICS

    error = y_test - predictions

    mae = np.mean(
        np.abs(error)
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(mse)

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (y_test - np.mean(y_test)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    # MODEL INFORMATION

    requested_epochs = epochs

    actual_epochs = getattr(
        model,
        "n_epochs_used",
        len(model.loss_history)
    )

    parameter_count = (
        len(model.weights) + 1
    )

    initial_loss = (
        model.loss_history[0]
        if len(model.loss_history) > 0
        else np.nan
    )

    final_loss = (
        model.loss_history[-1]
        if len(model.loss_history) > 0
        else np.nan
    )

    if actual_epochs < requested_epochs:
        training_status = "Early stopping activated"
    else:
        training_status = "Maximum epochs reached"

    # RESULTS TEXT

    results = f"""
MODEL TRAINING RESULTS
────────────────────────────────────────

Model                  : {model_name}
Learning Rate          : {learning_rate}
Requested Epochs       : {requested_epochs}
Actual Epochs Used     : {actual_epochs}

Polynomial Features    : {polynomial_features}

Parameters             : {parameter_count}

Initial Loss           : {initial_loss:,.4f}
Final Loss             : {final_loss:,.4f}

Training Status        : {training_status}


TEST PERFORMANCE
────────────────────────────────────────

MAE                    : {mae:,.4f}
MSE                    : {mse:,.4f}
RMSE                   : {rmse:,.4f}
R²                     : {r2:.6f}

R² Percentage          : {r2 * 100:.2f}%


OPTIMIZATION
────────────────────────────────────────

Gradient Descent       : Batch Gradient Descent
Optimizer              : Custom Gradient Descent
Loss Function          : Mean Squared Error
"""

    # GRAPH 1
    # LOSS VS EPOCH

    fig_loss, ax_loss = plt.subplots(
        figsize=(12, 6)
    )

    ax_loss.plot(
        range(1, len(model.loss_history) + 1),
        model.loss_history,
        color=BLACK,
        linewidth=2
    )

    # Mark final loss
    if len(model.loss_history) > 0:

        ax_loss.scatter(
            len(model.loss_history),
            model.loss_history[-1],
            color=ACCENT,
            s=70,
            zorder=5
        )

    ax_loss.set_xlabel(
        "Epoch",
        fontsize=13,
        color=BLACK
    )

    ax_loss.set_ylabel(
        "MSE Loss",
        fontsize=13,
        color=BLACK
    )

    ax_loss.grid(
        True,
        alpha=0.15
    )

    ax_loss.tick_params(
        labelsize=11,
        colors=BLACK
    )

    for spine in ax_loss.spines.values():
        spine.set_color(LIGHT_GRAY)

    fig_loss.tight_layout()


    # GRAPH 2
    # ACTUAL VS PREDICTED
    

    fig_prediction, ax_prediction = plt.subplots(
        figsize=(12, 6)
    )

    ax_prediction.scatter(
        y_test,
        predictions,
        color=BLACK,
        alpha=0.55,
        s=28
    )

    min_value = min(
        np.min(y_test),
        np.min(predictions)
    )

    max_value = max(
        np.max(y_test),
        np.max(predictions)
    )

    ax_prediction.plot(
        [min_value, max_value],
        [min_value, max_value],
        color=ACCENT,
        linewidth=2
    )

    ax_prediction.set_xlabel(
        "Actual Values",
        fontsize=13,
        color=BLACK
    )

    ax_prediction.set_ylabel(
        "Predicted Values",
        fontsize=13,
        color=BLACK
    )

    ax_prediction.grid(
        True,
        alpha=0.15
    )

    ax_prediction.tick_params(
        labelsize=11,
        colors=BLACK
    )

    for spine in ax_prediction.spines.values():
        spine.set_color(LIGHT_GRAY)

    fig_prediction.tight_layout()


    # 
    # GRAPH 3
    # PARAMETER LEARNING
    #
    # IMPORTANT:
    # KEEPING THE GRAPH LARGE
    # figsize = (13, 7)

    fig_parameters, ax_parameters = plt.subplots(
        figsize=(13, 7)
    )

    weights_history = getattr(
        model,
        "weights_history",
        []
    )

    bias_history = getattr(
        model,
        "bias_history",
        []
    )

    epochs_range = range(
        1,
        len(weights_history) + 1
    )

    # Plot weights

    if len(weights_history) > 0:

        weights_array = np.array(
            weights_history
        )

        # Make sure shape is correct
        if weights_array.ndim == 1:

            weights_array = (
                weights_array.reshape(-1, 1)
            )

        number_of_weights = (
            weights_array.shape[1]
        )

        for i in range(
            number_of_weights
        ):

            color = PARAMETER_COLORS[
                i % len(PARAMETER_COLORS)
            ]

            ax_parameters.plot(
                epochs_range,
                weights_array[:, i],
                color=color,
                linewidth=2,
                label=f"Weight {i + 1}"
            )

    
    # Plot bias
    

    if len(bias_history) > 0:

        ax_parameters.plot(
            range(
                1,
                len(bias_history) + 1
            ),
            bias_history,
            color="#333333",
            linewidth=2.5,
            linestyle="--",
            label="Bias"
        )

    ax_parameters.set_xlabel(
        "Epoch",
        fontsize=13,
        color=BLACK
    )

    ax_parameters.set_ylabel(
        "Parameter Value",
        fontsize=13,
        color=BLACK
    )

    ax_parameters.grid(
        True,
        alpha=0.15
    )

    ax_parameters.tick_params(
        labelsize=11,
        colors=BLACK
    )

    ax_parameters.legend(
        frameon=False,
        fontsize=10
    )

    for spine in ax_parameters.spines.values():
        spine.set_color(LIGHT_GRAY)

    # DO NOT REDUCE THIS GRAPH SIZE
    fig_parameters.tight_layout()


    
    # RETURN

    return (
        results,
        fig_loss,
        fig_prediction,
        fig_parameters
    )


# CUSTOM CSS

custom_css = """

/* ============================================================
   GLOBAL
============================================================ */

body {
    background: #FFFFFF !important;
    color: #181818 !important;
    font-family: Arial, Helvetica, sans-serif;
}

.gradio-container {
    background: #FFFFFF !important;
    max-width: 1400px !important;
}


/* ============================================================
   HERO
============================================================ */

.hero {
    padding: 80px 60px 70px 60px;
    background: #FFFFFF;
    border-bottom: 1px solid #D9D9D9;
}

.hero-logo {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #181818 !important;
    margin-bottom: 55px;
}

.hero-number {
    font-size: 13px;
    letter-spacing: 2px;
    color: #777777 !important;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 76px;
    line-height: 0.95;
    font-weight: 500;
    letter-spacing: -3px;
    color: #181818 !important;
}

/* GRADIENT now uses same dark shade */
.hero-title span {
    color: #181818 !important;
    font-weight: 700;
}


/* ============================================================
   SECTION HEADERS
============================================================ */

.section-header {
    background: #181818 !important;
    color: #FFFFFF !important;
    padding: 22px 26px;
    margin-top: 70px;
    margin-bottom: 35px;
}

.section-header h2 {
    color: #FFFFFF !important;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
}

.section-header h2 *,
.section-header p *,
.section-header span * {
    color: #FFFFFF !important;
}

.section-number {
    color: #FFFFFF !important;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    margin-bottom: 7px;
}


/* ============================================================
   MODEL CONFIGURATION
============================================================ */

.config-title {
    background: #181818 !important;
    color: #FFFFFF !important;
    padding: 20px 24px;
    font-size: 30px;
    font-weight: 700;
}

.config-title * {
    color: #FFFFFF !important;
}


/* ============================================================
   LABELS
============================================================ */

label {
    color: #181818 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

.gr-label {
    color: #181818 !important;
}


/* ============================================================
   DARK / GREY BLOCKS
============================================================ */

.dark-block,
.summary-block,
.section-dark {
    background: #181818 !important;
    color: #FFFFFF !important;
}

.dark-block *,
.summary-block *,
.section-dark * {
    color: #FFFFFF !important;
}


/* ============================================================
   TRAINING RESULTS
============================================================ */

.training-results {
    background: #FFFFFF !important;
    border: 1px solid #D9D9D9;
    padding: 35px;
}

.results-label {
    display: inline-block;
    background: #181818 !important;
    color: #FFFFFF !important;
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 18px;
}

.results-label *,
.results-label span,
.results-label p {
    color: #FFFFFF !important;
}

.results-box {
    background: #FFFFFF !important;
    color: #181818 !important;
    border: 1px solid #D9D9D9 !important;
}


/* ============================================================
   GRAPH SECTIONS
============================================================ */

.graph-section {
    margin-bottom: 115px;
}

.graph-heading {
    font-size: 30px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    color: #181818 !important;
    margin-bottom: 12px;
}

.graph-description {
    font-size: 16px;
    color: #777777 !important;
    line-height: 1.6;
    margin-bottom: 25px;
}


/* ============================================================
   GRAPH CONTAINERS
============================================================ */

.plot-container {
    min-height: 600px;
    background: #FFFFFF !important;
}


/* ============================================================
   BUTTON
============================================================ */

button {
    background: #181818 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

button *,
button span {
    color: #FFFFFF !important;
}

button:hover {
    background: #303030 !important;
    color: #FFFFFF !important;
}


/* ============================================================
   FOOTER
============================================================ */

.footer {
    border-top: 1px solid #D9D9D9;
    padding: 35px 0;
    margin-top: 80px;
    color: #777777 !important;
    font-size: 13px;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .hero {
        padding: 55px 25px;
    }

    .hero-title {
        font-size: 48px;
    }

    .section-header h2 {
        font-size: 22px;
    }

    .config-title {
        font-size: 24px;
    }

    .graph-heading {
        font-size: 24px !important;
    }

    .plot-container {
        min-height: 420px;
    }
}

"""


# GRADIO UI

with gr.Blocks(
    title="Gradient Descent Visualizer",
    css=custom_css
) as app:

    # HERO

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-logo">
                GD / VISUALIZER
            </div>

            <div class="hero-number">
                01 — MACHINE LEARNING
            </div>

            <div class="hero-title">
                UNDERSTAND<br>
                <span>GRADIENT</span><br>
                DESCENT.
            </div>

        </div>
        """
    )


    # MODEL CONFIGURATION

    gr.HTML(
        """
        <div class="section-header">

            <div class="section-number">
                02 — CONFIGURATION
            </div>

            <h2>
                MODEL CONFIGURATION
            </h2>

        </div>
        """
    )


    with gr.Row():

        with gr.Column():

            model_type = gr.Dropdown(
                choices=[
                    "Linear Regression",
                    "Polynomial Regression"
                ],
                value="Polynomial Regression",
                label="Model"
            )

        with gr.Column():

            learning_rate = gr.Number(
                value=0.01,
                label="Learning Rate"
            )

        with gr.Column():

            epochs = gr.Number(
                value=10000,
                label="Epochs"
            )

        with gr.Column():

            degree = gr.Number(
                value=2,
                label="Polynomial Degree"
            )


    train_button = gr.Button(
        "RUN GRADIENT DESCENT"
    )


    # TRAINING RESULTS

    gr.HTML(
        """
        <div class="section-header">

            <div class="section-number">
                03 — RESULTS
            </div>

            <h2>
                TRAINING RESULTS
            </h2>

        </div>
        """
    )


    results_output = gr.Textbox(
        label="RESULT SUMMARY",
        lines=25,
        interactive=False,
        elem_classes=["results-box"]
    )


    # LOSS GRAPH

    gr.HTML(
        """
        <div class="graph-section">

            <div class="graph-heading">
                LOSS VS EPOCH
            </div>

            <div class="graph-description">
                Observe how the Mean Squared Error decreases
                during gradient descent optimization.
            </div>

        </div>
        """
    )


    loss_plot = gr.Plot(
        show_label=False,
        elem_classes=["plot-container"]
    )


    # ACTUAL VS PREDICTED

    gr.HTML(
        """
        <div class="graph-section">

            <div class="graph-heading">
                ACTUAL VS PREDICTED
            </div>

            <div class="graph-description">
                Compare the actual insurance charges with
                the values predicted by the trained model.
            </div>

        </div>
        """
    )


    prediction_plot = gr.Plot(
        show_label=False,
        elem_classes=["plot-container"]
    )


    # PARAMETER LEARNING
    

    gr.HTML(
        """
        <div class="graph-section">

            <div class="graph-heading">
                PARAMETER LEARNING
            </div>

            <div class="graph-description">
                Track how model weights and bias change
                throughout the gradient descent process.
            </div>

        </div>
        """
    )


    parameters_plot = gr.Plot(
        show_label=False,
        elem_classes=["plot-container"]
    )


    # FOOTER

    gr.HTML(
        """
        <div class="footer">
            Gradient Descent Visualizer · Custom Gradient Descent
        </div>
        """
    )


    # BUTTON ACTION

    train_button.click(
        fn=train_model,
        inputs=[
            model_type,
            learning_rate,
            epochs,
            degree
        ],
        outputs=[
            results_output,
            loss_plot,
            prediction_plot,
            parameters_plot
        ]
    )


# RUN APPLICATION

if __name__ == "__main__":

    app.launch()
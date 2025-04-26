import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils import (
    load_soh_per_battery,
    preprocess_data,
    train_and_evaluate,
    generate_plots,
    extract_capacity_and_predictions
)
from models import TransformerSOHPredictor, LSTM_CNN_SOHPredictor

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "prediction": None,
        "metrics": None,
        "soh_plot": None,
        "rul_plot": None,
        "hist_plot": None,
        "heatmap_plot": None,
        "table_plot": None,
        "capacity_image": None,
        "prediction_table": None,
        "selected_file": None,
        "model_type": None,
    }

    if request.method == "POST":
        model_type = request.form.get("model")
        uploaded_file = request.files.get("uploaded_file")
        selected_file = request.form.get("dataset")

        if uploaded_file and uploaded_file.filename != "":
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(filepath)
            data_path = filepath
            context["selected_file"] = filename
        elif selected_file:
            data_path = os.path.join("data", selected_file)
            context["selected_file"] = selected_file
        else:
            return render_template("index.html", error="Please select or upload a dataset.")

        soh_dict = load_soh_per_battery([data_path])
        all_soh = list(soh_dict.values())[0]
        X_tensor, y_tensor, dataloader, soh_min, soh_max = preprocess_data(all_soh)

        model = TransformerSOHPredictor(seq_len=20) if model_type == "Transformer" else LSTM_CNN_SOHPredictor(seq_len=20)
        prediction, true_soh, metrics = train_and_evaluate(model, dataloader, X_tensor, y_tensor, soh_min, soh_max)
        soh_plot, rul_plot, hist_plot, heatmap_plot, table_plot = generate_plots(true_soh, prediction)

        capacity_image, prediction_table = extract_capacity_and_predictions(model, X_tensor, y_tensor)

        context.update({
            "prediction": prediction,
            "metrics": metrics,
            "model_type": model_type,
            "soh_plot": soh_plot,
            "rul_plot": rul_plot,
            "hist_plot": hist_plot,
            "heatmap_plot": heatmap_plot,
            "table_plot": table_plot,
            "capacity_image": capacity_image,
            "prediction_table": prediction_table,
        })

    return render_template("index.html", **context)

if __name__ == "__main__":
    app.run(debug=True)

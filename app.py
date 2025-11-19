from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import uvicorn
from src.serving.serving import DeployServe
# from src.monitoring.monitoring import MLMonitor
# from monitor_scheduler import start_monitoring_scheduler
import os
from datetime import datetime
# from utils.results import Alert_Severity # Added import

app = Flask(__name__)

serving = DeployServe()
# monitor = MLMonitor()

# Start the monitoring scheduler in a background thread
# start_monitoring_scheduler(monitor, interval_seconds=30)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", active_mode="dash")


@app.route("/predict_single", methods=["POST"])
def predict_single():
    text_input = request.form.get("feature1", "")
    if not text_input:
        return render_template(
            "index.html",
            prediction="No input text",
            active_mode="single",
            feature1=text_input,
        )

    try:
        pred = serving.single_predict(text_input)
        # print(f"{pred}:{text_input}")
        return render_template(
            "index.html",
            prediction=pred,
            active_mode="single",
            feature1=text_input,
        )
    except Exception as e:
        return render_template(
            "index.html",
            result=f"Prediction failed: {e}",
            active_mode="single",
            feature1=text_input,
        )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if "file" not in request.files:
        return render_template(
            "index.html", batch_result="No file uploaded", active_mode="batch"
        )

    file = request.files["file"]
    upload_dir = "user_app/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    try:
        # results_df = serving.batch_prediction(file_path)
        results_df = serving.batch_predict(file_path)
        table_html = results_df.to_html(classes="table table-striped", index=False)
        return render_template(
            "index.html",
            batch_result=table_html,
            active_mode="batch",
            batch_filename=file.filename,
        )
    except Exception as e:
        return render_template(
            "index.html", batch_result=f"Batch prediction failed: {e}", active_mode="batch"
        )
        

@app.route("/history", methods=["GET"])
def history():
    mode = request.args.get("mode", "all")          # single/batch/all
    sort_order = request.args.get("sort", "desc")   # asc/desc

    df = serving.get_history(None if mode=="all" else mode)
    if df is not None and not df.empty:
        if "timestamp" in df.columns:  # assuming timestamp column exists
            df = df.sort_values("timestamp", ascending=(sort_order=="asc"))
        history_table = df.to_html(classes="table table-striped", index=False)
    else:
        history_table = None

    return render_template(
        "index.html", 
        history_table=history_table, 
        active_mode="history",
        current_mode=mode,
        sort_order=sort_order
    )


@app.route("/predictions_csv", methods=["GET"]) 
def download_csv():
    if not os.path.exists(serving.HISTORY_FILE):
        return "No history available", 404

    # Create filename timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.csv"

    return send_file(
        serving.HISTORY_FILE,
        as_attachment=True,
        download_name=filename,  # dynamic filename
        mimetype='text/csv'
    )

# @app.route("/monitor/metrics", methods=["GET"])
# def metrics():
#     return jsonify(monitor.get_system_metrics())

# @app.route("/monitor/drift", methods=["GET"])
# def drift():
#     return jsonify(monitor.check_drift())

# @app.route("/monitor/performance", methods=["GET"])
# def performance():
#     return jsonify(monitor.model_performance())

# @app.route("/monitor/health", methods=["GET"])
# def health():
#     return jsonify(monitor.system_health())
# @app.route('/monitor/metrics', methods=["GET"])
# def get_metrics():
#     result = monitor.get_system_metrics()
#     # return result.to_json()
#     return jsonify(result.to_dict())

# @app.route('/monitor/check_drift', methods=["GET"])
# def check_drift():
#     result = monitor.check_drift()
#     # return result.to_json()
#     return jsonify(result.to_dict())


# @app.route("/monitor/request_approval", methods=["POST"])
# def monitor_request_approval():
#     # Example: create an alert to notify team
#     alert = monitor.create_alert(
#         alert_type="APPROVAL_REQUEST",
#         severity=Alert_Severity.ALERT_INFO,
#         message="User requested manual approval",
#         details={"timestamp": datetime.now().isoformat()},
#     )
#     # return json.dumps({"status": "success", "alert": alert})
#     return jsonify({"status": "success", "alert": alert})



# @app.route("/debug", methods=["GET"])
# def debug():
#     try:
#         return {
#             "model_loaded": serving.pipeline is not None,
#             "has_deployment_info": bool(serving.deployment_info),
#             "current_run_id": serving.current_run_id,
#             "history_file_exists": os.path.exists(serving.HISTORY_FILE),
#         }
#     except Exception as e:
#         return {"error": str(e)}


if __name__ == "__main__":
    app.run(debug=True, port=9000)
    # uvicorn.run(app, host="0.0.0.0", port=8000)

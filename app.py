from flask import Flask, render_template, request, send_file
from generate_text import generate_text
import time
import datetime
import os

app = Flask(__name__)
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    generation_time = None
    prompt = ""
    file_name = None

    if request.method == "POST":
        prompt = request.form["prompt"]
        model_name = request.form.get("model_name", "gpt2")
        max_length = int(request.form.get("max_length", 250))
        temperature = float(request.form.get("temperature", 0.7))
        top_k = int(request.form.get("top_k", 50))
        top_p = float(request.form.get("top_p", 0.95))

        start_time = time.time()
        generated_text = generate_text(
            prompt=prompt,
            model_name=model_name,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        end_time = time.time()
        generation_time = round(end_time - start_time, 2)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{output_folder}/generated_output_{timestamp}.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt}\n\nGenerated Text:\n{generated_text}\n")

    return render_template("index.html", prompt=prompt, generated_text=generated_text, generation_time=generation_time, file_name=file_name)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

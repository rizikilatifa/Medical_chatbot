print("Starting minimal test...", flush=True)

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello! Flask is working."

print("Server starting on http://localhost:8080", flush=True)
app.run(host="0.0.0.0", port=8080, debug=True)

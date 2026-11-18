"""Flask route for the last ten CareOps outcomes."""

import httpx
from flask import Flask, jsonify

from .summary import fetch_summary

app = Flask(__name__)


@app.get("/summary")
def summary():
    try:
        return jsonify(fetch_summary())
    except (httpx.HTTPError, TypeError, ValueError, KeyError):
        return jsonify({"detail": "CareOps API unavailable"}), 502

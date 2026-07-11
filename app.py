from flask import Flask, jsonify, request
from flask_cors import CORS

from model_service import (
    analyze_reviews,
    get_sentiment_analyzer,
    predict_batch_texts,
    predict_single,
)

app = Flask(__name__)
CORS(app)

# Warm up model on startup
get_sentiment_analyzer()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})


@app.route('/analyze', methods=['POST'])
def analyze():
    """Review insights dashboard (Laravel AIReviewService)."""
    data = request.get_json(silent=True) or {}
    reviews = data.get('reviews', [])
    insights = analyze_reviews(reviews)
    return jsonify({'success': True, 'data': insights})


@app.route('/predict', methods=['POST'])
def predict():
    """Single review sentiment (Laravel AiSentimentService)."""
    data = request.get_json(silent=True) or {}
    review_text = data.get('review', '')
    if not review_text:
        return jsonify({'error': 'review text is required'}), 400

    return jsonify(predict_single(review_text))


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch sentiment for review texts (Laravel AiSentimentService)."""
    data = request.get_json(silent=True) or {}
    reviews = data.get('reviews', [])

    if not isinstance(reviews, list):
        return jsonify({'error': 'reviews must be an array'}), 400

    return jsonify(predict_batch_texts(reviews))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)

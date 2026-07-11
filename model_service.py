"""AI model loading and review analysis for NexVex Review Insights."""

import re
from collections import Counter

from transformers import pipeline

POSITIVE_KEYWORDS = [
    'great', 'excellent', 'amazing', 'professional', 'fast', 'clean',
    'friendly', 'reliable', 'perfect', 'awesome',
]
NEGATIVE_KEYWORDS = [
    'terrible', 'bad', 'poor', 'slow', 'late', 'rude', 'expensive',
    'worst', 'horrible', 'useless',
]
STOPWORDS = {
    'the', 'and', 'for', 'that', 'this', 'with', 'was', 'were', 'have',
    'has', 'had', 'but', 'not', 'are', 'all', 'can', 'get', 'from',
    'they', 'will', 'you', 'your', 'our',
}

_sentiment_analyzer = None


def get_sentiment_analyzer():
    """Load Hugging Face sentiment model once."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
        )
    return _sentiment_analyzer


def _empty_insights():
    return {
        'customers': 0,
        'total_reviews': 0,
        'sentiment': {
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'positive_percentage': 0,
        },
        'reorder_probability': 0,
        'common_praise': [],
        'common_complaints': [],
        'ai_summary': 'No reviews yet.',
    }


def _extract_keywords(comments, limit=5):
    all_text = ' '.join(comments).lower()
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    filtered = [w for w in words if w not in STOPWORDS]
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(limit)]


def _keyword_fallback(text):
    text_lower = text.lower()
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    if pos_count > neg_count:
        return 'positive'
    if neg_count > pos_count:
        return 'negative'
    return 'neutral'


def _analyze_text(text, analyzer):
    text = (text or '').strip()
    if not text:
        return 'neutral'

    try:
        result = analyzer(text[:512])[0]
        if result['label'] == 'POSITIVE':
            return 'positive'
        return 'negative'
    except Exception:
        return _keyword_fallback(text)


def _predict_text_details(text, analyzer):
    """Return Laravel-compatible sentiment payload for a single review text."""
    text = (text or '').strip()
    if not text:
        return {'sentiment': 'Neutral', 'confidence': 0.0, 'keywords': []}

    try:
        result = analyzer(text[:512])[0]
        label = result['label']
        score = float(result['score'])
        sentiment = 'Positive' if label == 'POSITIVE' else 'Negative'
        return {
            'sentiment': sentiment,
            'confidence': round(score, 4),
            'keywords': _extract_keywords([text], 5),
        }
    except Exception:
        fallback = _keyword_fallback(text)
        if fallback == 'positive':
            return {
                'sentiment': 'Positive',
                'confidence': 0.5,
                'keywords': _extract_keywords([text], 5),
            }
        if fallback == 'negative':
            return {
                'sentiment': 'Negative',
                'confidence': 0.5,
                'keywords': _extract_keywords([text], 5),
            }
        return {'sentiment': 'Neutral', 'confidence': 0.5, 'keywords': []}


def predict_single(review_text):
    """Single review prediction for /predict."""
    analyzer = get_sentiment_analyzer()
    details = _predict_text_details(review_text, analyzer)
    return {
        'sentiment': details['sentiment'],
        'confidence': details['confidence'],
        'keywords': details['keywords'],
    }


def predict_batch_texts(review_texts):
    """Batch prediction for /predict/batch (Laravel AiSentimentService)."""
    analyzer = get_sentiment_analyzer()
    all_results = []
    positive = 0
    negative = 0

    for index, raw in enumerate(review_texts, start=1):
        if isinstance(raw, dict):
            text = (raw.get('comment') or raw.get('text') or '').strip()
        else:
            text = str(raw or '').strip()

        details = _predict_text_details(text, analyzer)
        if details['sentiment'] == 'Positive':
            positive += 1
        elif details['sentiment'] == 'Negative':
            negative += 1

        all_results.append({
            'index': index,
            'review': text,
            'sentiment': details['sentiment'],
            'confidence': details['confidence'],
            'keywords': details['keywords'],
        })

    return {
        'all_results': all_results,
        'summary': {
            'predicted_positive': positive,
            'predicted_negative': negative,
            'total': len(all_results),
        },
    }


def _build_summary(total, positive_percentage, common_praise, common_complaints):
    if total == 0:
        return 'No reviews yet. Encourage customers to leave feedback.'

    praise = ', '.join(common_praise[:3]) or 'quality service'
    complaints = ', '.join(common_complaints[:3]) or 'service issues'

    if positive_percentage >= 80:
        return (
            f'Excellent! {positive_percentage}% of customers are satisfied. '
            f'Customers frequently mention: {praise}. Keep up the great work!'
        )
    if positive_percentage >= 60:
        return (
            f'Good! {positive_percentage}% customer satisfaction. '
            f'Common praise: {praise}. Focus on improving {100 - positive_percentage}% of feedback.'
        )
    if positive_percentage >= 40:
        return (
            f'Mixed reviews. {positive_percentage}% positive. '
            f'Common concerns: {complaints}. Consider addressing these issues.'
        )
    return (
        f'Needs improvement. Only {positive_percentage}% positive feedback. '
        f'Main complaints: {complaints}. Review customer feedback carefully.'
    )


def analyze_reviews(reviews):
    """Analyze provider reviews and return insight payload."""
    if not reviews:
        return _empty_insights()

    analyzer = get_sentiment_analyzer()
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    positive_comments = []
    negative_comments = []

    for review in reviews:
        text = review.get('comment', '') or ''
        if not text.strip():
            rating = review.get('rating')
            if rating is not None:
                text = f'Customer gave {rating} out of 5 stars.'
            else:
                neutral_count += 1
                continue

        sentiment = _analyze_text(text, analyzer)
        if sentiment == 'positive':
            positive_count += 1
            positive_comments.append(text)
        elif sentiment == 'negative':
            negative_count += 1
            negative_comments.append(text)
        else:
            neutral_count += 1

    total = len(reviews)
    positive_percentage = round((positive_count / total) * 100) if total > 0 else 0
    common_praise = _extract_keywords(positive_comments, 5)
    common_complaints = _extract_keywords(negative_comments, 5)
    reorder_probability = positive_percentage

    customer_ids = {
        r.get('client_id') or r.get('customer_id')
        for r in reviews
        if r.get('client_id') or r.get('customer_id')
    }

    return {
        'customers': len(customer_ids),
        'total_reviews': total,
        'sentiment': {
            'positive': positive_count,
            'neutral': neutral_count,
            'negative': negative_count,
            'positive_percentage': positive_percentage,
        },
        'reorder_probability': reorder_probability,
        'common_praise': common_praise,
        'common_complaints': common_complaints,
        'ai_summary': _build_summary(
            total, positive_percentage, common_praise, common_complaints
        ),
    }

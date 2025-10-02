# Responsible AI Checklist: Mama Earth Sentiment Analysis

## Fairness
- Check model bias: Does it favor positive or negative reviews unfairly?
- Analyze performance across different product categories or demographics.

## Privacy
- Do not store personal identifiers in reviews.
- Ensure data anonymization if collecting new data.

## Consent & Transparency
- Only analyze reviews with explicit consent.
- Explain model predictions to end users via SHAP visualizations.

## Security & Robustness
- Prevent injection attacks in user input.
- Validate input before prediction.

## Monitoring & Maintenance
- Monitor model drift (reviews might change over time).
- Retrain model periodically with updated reviews.

from model import FraudDetector
from app import DomainVerifier


def test_fraud_detector_prediction_labels():
    detector = FraudDetector()
    label_good, conf_good = detector.predict(
        "Google internship posted on careers.google.com contact talent@google.com"
    )
    label_bad, conf_bad = detector.predict(
        "No interview job offer, pay registration fee now and contact urgentjob@gmail.com"
    )

    assert label_good in {"Genuine", "Fraudulent"}
    assert label_bad in {"Genuine", "Fraudulent"}
    assert 0.0 <= conf_good <= 1.0
    assert 0.0 <= conf_bad <= 1.0


def test_domain_verifier_flags_free_domain():
    verifier = DomainVerifier()
    result = verifier.verify("Meta internship HR support is hrmeta@gmail.com")

    assert result.status in {"Suspicious", "Mixed"}
    assert "free email provider" in result.details

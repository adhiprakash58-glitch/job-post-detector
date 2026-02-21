from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class Prediction:
    label: str
    confidence: float


class FraudDetector:
    def __init__(self) -> None:
        self.pipeline = self._train_model()

    def _load_data(self) -> pd.DataFrame:
        data = [
            ("Google is hiring software engineering interns. Apply via careers.google.com and contact internrecruiting@google.com", 0),
            ("Microsoft internship opportunity with stipend and interview process. Reach us at university@microsoft.com", 0),
            ("Urgent! Earn 5000 USD weekly from home no skills required. send details to hrjoboffer456@gmail.com", 1),
            ("Congrats you are selected without interview pay registration fee and confirm at offersdesk@yahoo.com", 1),
            ("Amazon cloud support role posted on official portal. Contact talent@amazon.com for queries", 0),
            ("Data entry job for students. Pay processing charge now and email cv to fastcashjob@outlook.com", 1),
            ("Meta summer internship 2026 with hybrid work mode apply at careers.meta.com", 0),
            ("Guaranteed placement in MNC, limited seats, message with aadhaar and passport to quickhire@protonmail.com", 1),
            ("TCS digital internship program with coding round details shared from tcs.careers@tcs.com", 0),
            ("No interview required and instant offer letter download. share OTP for verification to get joining", 1),
            ("Wipro analyst opening with legitimate HR discussion round and mail from recruitment@wipro.com", 0),
            ("LinkedIn remote internship post asking to purchase training kit before onboarding", 1),
        ]
        return pd.DataFrame(data, columns=["text", "label"])

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9@\.\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _train_model(self) -> Pipeline:
        df = self._load_data()
        df["text"] = df["text"].map(self._normalize)

        X_train, X_test, y_train, _ = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )

        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        pipeline.fit(X_train, y_train)
        return pipeline

    def predict(self, text: str) -> tuple[str, float]:
        cleaned = self._normalize(text)
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        fraud_probability = float(probabilities[1])
        label = "Fraudulent" if fraud_probability >= 0.5 else "Genuine"
        confidence = fraud_probability if label == "Fraudulent" else 1 - fraud_probability
        return label, confidence

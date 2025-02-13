import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateBugReportDetector:
    def __init__(self, threshold=0.8):
        """
        Initialize the duplicate detection model.
        :param threshold: Similarity threshold to classify duplicate reports.
        """
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.report_vectors = None
        self.report_ids = None

    def fit(self, bug_reports, report_ids):
        """
        Train the model with existing bug reports.
        :param bug_reports: List of bug report texts.
        :param report_ids: Corresponding bug report IDs.
        """
        self.report_vectors = self.vectorizer.fit_transform(bug_reports)
        self.report_ids = report_ids

    def predict(self, new_report):
        """
        Predict whether a new bug report is a duplicate.
        :param new_report: Text of the new bug report.
        :return: (is_duplicate, duplicate_id) where is_duplicate is a boolean and duplicate_id is the matched report ID.
        """
        if self.report_vectors is None:
            raise ValueError("Model has not been trained. Call fit() first.")

        new_vector = self.vectorizer.transform([new_report])
        similarities = cosine_similarity(new_vector, self.report_vectors).flatten()

        max_sim_index = np.argmax(similarities)
        max_sim_value = similarities[max_sim_index]

        if max_sim_value >= self.threshold:
            return True, self.report_ids[max_sim_index]
        else:
            return False, None


def preprocess_bug_reports(csv_path):
    """
    Load and preprocess bug report dataset.
    :param csv_path: Path to the bug report CSV file.
    :return: Lists of report IDs and corresponding texts.
    """
    data = pd.read_csv(csv_path, encoding='utf-8')
    report_ids = data['BID'].tolist()
    report_texts = (data['summary'] + " " + data['description']).fillna("").tolist()
    return report_ids, report_texts


def filter_non_duplicate_reports(csv_path, detector):
    """
    Filter out duplicate bug reports before classification.
    :param csv_path: Path to bug report CSV file.
    :param detector: Trained DuplicateBugReportDetector instance.
    :return: List of non-duplicate bug reports.
    """
    report_ids, report_texts = preprocess_bug_reports(csv_path)
    non_duplicate_reports = []

    for i, text in enumerate(report_texts):
        is_duplicate, dup_id = detector.predict(text)
        if not is_duplicate:
            non_duplicate_reports.append((report_ids[i], text))

    return non_duplicate_reports


if __name__ == "__main__":
    csv_path = "../data/netbeans/netbeans.csv"  # Adjust path as needed

    # Load dataset and train duplicate detection model
    report_ids, report_texts = preprocess_bug_reports(csv_path)
    detector = DuplicateBugReportDetector(threshold=0.8)
    detector.fit(report_texts, report_ids)

    # Filter non-duplicate reports
    non_duplicate_reports = filter_non_duplicate_reports(csv_path, detector)

    print(f"Filtered {len(non_duplicate_reports)} non-duplicate bug reports for classification.")

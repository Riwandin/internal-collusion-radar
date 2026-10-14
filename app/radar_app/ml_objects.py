"""Objek kecil yang ikut tersimpan di file model (joblib).

Kenapa di modul terpisah? File joblib menyimpan *referensi* ke kelas, bukan kodenya. Kalau kelas ini hanya didefinisikan
di notebook, aplikasi Streamlit tidak bisa membuka model. Dengan menaruhnya di sini, notebook (training) dan aplikasi
(scoring) memakai kelas yang sama persis.

Isi:
- SimpleSMOTE            : SMOTE sederhana (dipakai hanya kalau paket imbalanced-learn tidak terpasang)
- ResamplingClassifier   : preprocessing -> resampling (SMOTE) -> model, resampling HANYA saat fit (data latih)
- CalibratedRiskModel    : model + kalibrasi isotonic + threshold -> skor risiko 0-100 yang bisa dipertanggungjawabkan
- MultiViewRiskEngine    : 4 pilar (Rules, Isolation Forest, Classifier konten, Model relasi/graph) digabung dengan
                           stacking logistic regression yang bobotnya DIPELAJARI di validasi 2025 (late fusion)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors


class SimpleSMOTE:
    """SMOTE (Chawla et al., 2002): buat titik sintetis di antara satu kasus minoritas dan tetangga terdekatnya."""

    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        minority = X[y == 1]
        n_needed = int((y == 0).sum() - (y == 1).sum())
        if n_needed <= 0 or len(minority) <= self.k_neighbors:
            return X, y
        neighbors = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(minority)
        _, neighbor_index = neighbors.kneighbors(minority)
        base = rng.integers(0, len(minority), n_needed)
        partner = neighbor_index[base, rng.integers(1, self.k_neighbors + 1, n_needed)]
        gap = rng.random((n_needed, 1))
        synthetic = minority[base] + gap * (minority[partner] - minority[base])
        return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_needed, dtype=int)])


class ResamplingClassifier(ClassifierMixin, BaseEstimator):
    """preprocessor -> sampler.fit_resample (hanya saat fit) -> estimator.

    Setara `imblearn.pipeline.Pipeline`, tapi tidak butuh paket tambahan. Saat predict, TIDAK ada resampling:
    data validasi/test/produksi dinilai apa adanya.
    """

    def __init__(self, preprocessor=None, sampler=None, estimator=None):
        self.preprocessor = preprocessor
        self.sampler = sampler
        self.estimator = estimator

    def fit(self, X, y):
        self.preprocessor_ = clone(self.preprocessor).fit(X, y)
        X_transformed = self.preprocessor_.transform(X)
        X_resampled, y_resampled = self.sampler.fit_resample(X_transformed, np.asarray(y))
        self.estimator_ = clone(self.estimator).fit(X_resampled, y_resampled)
        self.classes_ = self.estimator_.classes_
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.preprocessor_.transform(X))

    def predict(self, X):
        return self.estimator_.predict(self.preprocessor_.transform(X))


def split_preprocessor_and_estimator(model):
    """Ambil (preprocessor terlatih, estimator terlatih) dari Pipeline sklearn/imblearn atau ResamplingClassifier."""
    if isinstance(model, ResamplingClassifier):
        return model.preprocessor_, model.estimator_
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        return (steps[0] if len(steps) > 1 else None), steps[-1]
    return None, model


class CalibratedRiskModel:
    """Skor akhir transaksi = probabilitas terkalibrasi (0-1), ditampilkan sebagai 0-100.

    - base_model   : pipeline terlatih (train 2021-2024)
    - calibrator   : IsotonicRegression yang dipetakan di data validasi 2025 (skor mentah -> peluang empiris)
    - threshold    : batas prioritas "Tinggi" (dipilih di validasi, pada skala terkalibrasi)
    Isotonic bersifat monoton, jadi URUTAN skor tidak berubah; hanya angkanya yang jadi bisa dibaca sebagai peluang.
    """

    def __init__(self, base_model, calibrator, threshold: float, features: list[str],
                 categorical_features: list[str], model_name: str, watch_ratio: float = 0.5):
        self.base_model = base_model
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.features = list(features)
        self.categorical_features = list(categorical_features)
        self.model_name = model_name
        self.watch_ratio = watch_ratio

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        matrix = df[self.features].copy()
        numeric = [f for f in self.features if f not in self.categorical_features]
        matrix[numeric] = matrix[numeric].astype(float)
        for column in self.categorical_features:
            matrix[column] = matrix[column].astype(object).where(matrix[column].notna(), np.nan)
        return matrix

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(self.prepare(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw_score(df)), 0, 1)

    def risk_band(self, probability) -> np.ndarray:
        probability = np.asarray(probability)
        return np.select([probability >= self.threshold, probability >= self.threshold * self.watch_ratio],
                         ["Tinggi", "Sedang"], "Rendah")

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        probability = self.predict_proba(df)
        return pd.DataFrame({"risk_probability": probability, "risk_score": np.round(probability * 100, 1),
                             "risk_band": self.risk_band(probability)}, index=df.index)


class NonNegativeLogisticRegression(ClassifierMixin, BaseEstimator):
    """Logistic regression L2 dengan koefisien terpilih dipaksa ≥ 0 (monoton).

    Objektif sama dengan sklearn LogisticRegression (0,5·‖w‖² + C·Σ log-loss, intercept tidak dipenalti), diselesaikan
    dengan L-BFGS-B ber-batas. Dipakai sebagai penggabung pilar: skor pilar yang naik tidak boleh menurunkan skor akhir,
    sehingga setiap kontribusi bisa dijelaskan searah dengan arti pilarnya.
    """

    def __init__(self, C: float = 1.0, nonnegative=None, max_iter: int = 2000):
        self.C = C
        self.nonnegative = nonnegative
        self.max_iter = max_iter

    def fit(self, X, y):
        from scipy.optimize import minimize
        from scipy.special import expit

        X = np.asarray(X, dtype=float)
        signs = 2 * np.asarray(y).astype(int) - 1
        n_features = X.shape[1]
        constrained = set(range(n_features)) if self.nonnegative is None else set(self.nonnegative)

        def objective(theta):
            weights, bias = theta[:n_features], theta[n_features]
            margin = signs * (X @ weights + bias)
            loss = np.logaddexp(0, -margin).sum()
            gradient_z = -signs * expit(-margin)
            gradient = np.append(weights + self.C * (X.T @ gradient_z), self.C * gradient_z.sum())
            return 0.5 * weights @ weights + self.C * loss, gradient

        bounds = [(0, None) if j in constrained else (None, None) for j in range(n_features)] + [(None, None)]
        result = minimize(objective, np.zeros(n_features + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": self.max_iter})
        self.coef_ = result.x[:n_features].reshape(1, -1)
        self.intercept_ = np.array([result.x[n_features]])
        self.classes_ = np.array([0, 1])
        self.n_iter_ = result.nit
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X) -> np.ndarray:
        probability = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - probability, probability])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


PILLAR_LABELS = {"rule": "Pilar 1 · Rules SOP", "iforest": "Pilar 2 · Anomali approver (IF)",
                 "classifier": "Pilar 3 · Konten transaksi", "graph": "Pilar 4 · Relasi & jaringan"}
RULE_COLUMNS = ["rf_near_limit", "rf_after_hours", "rf_fast_approval", "rf_debtor_repeat", "rf_pair_dependency"]


def _logit(probability) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(probability / (1 - probability))


class MultiViewRiskEngine:
    """Skor akhir transaksi dari 4 pilar yang fiturnya TIDAK tumpang tindih, digabung dengan stacking.

    Pilar (setiap kolom data mentah hanya masuk ke SATU pilar yang belajar dari label):
      1. rule       S_rule  = 20 × jumlah rule SOP terpenuhi (0–100); deterministik, tanpa label
      2. iforest    S_IF    = persentil anomali approver pada bulan transaksi (Isolation Forest, tanpa label, fitur
                              perilaku saja). Kosong kalau approver < 10 keputusan → nilai netral + indikator "kosong"
      3. classifier S_cls   = probabilitas model KONTEN transaksi (nominal, waktu, profil & riwayat debitur)
      4. graph      S_graph = probabilitas model RELASI (pasangan requester–approver, routing, fitur graph g_*)

    Fusion: logistic regression monoton di atas [logit(p_cls mentah), logit(p_graph mentah), S_rule/100, S_IF/100,
    IF_kosong] (+ opsional S_bukti_jaringan/100), dilatih di validasi 2025 (di sana skor pilar out-of-sample). p_* mentah
    = probabilitas model pilar SEBELUM kalibrasi isotonic (lebih halus, tanpa tangga); S_cls/S_graph terkalibrasi dipakai
    untuk tampilan dan jaring pengaman. Karena linear di skala log-odds, skor akhir bisa diurai PERSIS:
    logit(R) = dasar + Σ kontribusi pilar.

    `use_graph_evidence=True` (konfigurasi C) menambah skor bukti jaringan yang bisa dijelaskan (radar_app.hetero_graph)
    sebagai input kedua Pilar 4. Kontribusinya tetap dihitung ke Pilar 4, jadi tidak ada "pilar ke-5" tersembunyi.
    """

    STACK_COLUMNS = ["logit_cls", "logit_graph", "rule_scaled", "if_scaled", "if_missing"]
    EVIDENCE_COLUMN = "graph_evidence_scaled"
    PILLAR_OF_COLUMN = {"logit_cls": "classifier", "logit_graph": "graph", "rule_scaled": "rule",
                        "if_scaled": "iforest", "if_missing": "iforest", "graph_evidence_scaled": "graph"}

    def __init__(self, classifier_model: CalibratedRiskModel, graph_model: CalibratedRiskModel, stacker,
                 reference_means: dict, threshold: float, watch_threshold: float, escalation_thresholds: dict,
                 model_name: str = "Multi-View Risk Engine (stacking 4 pilar)", if_neutral: float = 50.0,
                 use_graph_evidence: bool = False):
        self.classifier_model = classifier_model
        self.graph_model = graph_model
        self.stacker = stacker
        self.reference_means = dict(reference_means)
        self.threshold = float(threshold)
        self.watch_threshold = float(watch_threshold)
        self.escalation_thresholds = dict(escalation_thresholds)
        self.model_name = model_name
        self.if_neutral = float(if_neutral)
        self.use_graph_evidence = bool(use_graph_evidence)

    @property
    def stack_columns(self) -> list[str]:
        return self.STACK_COLUMNS + ([self.EVIDENCE_COLUMN] if getattr(self, "use_graph_evidence", False) else [])

    # ------------------------------------------------------------------ input yang dibutuhkan
    @property
    def features(self) -> list[str]:
        """Semua kolom input yang dibaca engine (untuk form live test / validasi input)."""
        extra = RULE_COLUMNS + ["rule_score", "approver_anomaly_score"]
        if getattr(self, "use_graph_evidence", False):
            from radar_app.hetero_graph import EVIDENCE_COLUMNS
            extra = extra + EVIDENCE_COLUMNS
        return list(dict.fromkeys(self.classifier_model.features + self.graph_model.features + extra))

    @property
    def categorical_features(self) -> list[str]:
        return list(dict.fromkeys(self.classifier_model.categorical_features + self.graph_model.categorical_features))

    # ------------------------------------------------------------------ skor per pilar
    def pillar_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        if "rule_score" in df:
            rule_count = pd.to_numeric(df["rule_score"], errors="coerce")
        else:
            rule_count = df[[c for c in RULE_COLUMNS if c in df]].astype(float).sum(axis=1)
        if "approver_anomaly_score" in df:
            if_score = pd.to_numeric(df["approver_anomaly_score"], errors="coerce")
        else:
            if_score = pd.Series(np.nan, index=df.index)
        cls_raw = self.classifier_model.raw_score(df)
        graph_raw = self.graph_model.raw_score(df)
        scores = pd.DataFrame({
            "s_rule": np.clip(rule_count.fillna(0).to_numpy(dtype=float) * 20.0, 0, 100),
            "s_if": if_score.to_numpy(dtype=float),
            "if_missing": if_score.isna().astype(int).to_numpy(),
            "s_cls": np.round(np.clip(self.classifier_model.calibrator.predict(cls_raw), 0, 1) * 100, 1),
            "s_graph": np.round(np.clip(self.graph_model.calibrator.predict(graph_raw), 0, 1) * 100, 1),
            "p_cls_raw": cls_raw, "p_graph_raw": graph_raw}, index=df.index)
        if getattr(self, "use_graph_evidence", False):
            from radar_app.hetero_graph import graph_evidence_score
            scores["s_graph_evidence"] = graph_evidence_score(df).to_numpy()
        return scores

    def stack_matrix(self, scores: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.DataFrame({"logit_cls": _logit(scores["p_cls_raw"]), "logit_graph": _logit(scores["p_graph_raw"]),
                               "rule_scaled": scores["s_rule"] / 100.0,
                               "if_scaled": scores["s_if"].fillna(self.if_neutral) / 100.0,
                               "if_missing": scores["if_missing"].astype(float)}, index=scores.index)
        if getattr(self, "use_graph_evidence", False):
            matrix[self.EVIDENCE_COLUMN] = scores["s_graph_evidence"] / 100.0
        return matrix[self.stack_columns]

    # ------------------------------------------------------------------ skor akhir
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.stacker.predict_proba(self.stack_matrix(self.pillar_scores(df)).to_numpy())[:, 1]

    def contributions(self, df: pd.DataFrame, scores: pd.DataFrame | None = None) -> pd.DataFrame:
        """Kontribusi tiap pilar ke log-odds skor akhir, relatif terhadap rata-rata validasi (jumlahnya PERSIS)."""
        scores = self.pillar_scores(df) if scores is None else scores
        matrix = self.stack_matrix(scores)
        coefficients = dict(zip(self.stack_columns, self.stacker.coef_[0]))
        out = pd.DataFrame(index=df.index)
        for pillar in PILLAR_LABELS:
            columns = [c for c in self.stack_columns if self.PILLAR_OF_COLUMN[c] == pillar]
            out[f"c_{pillar}"] = sum(coefficients[c] * (matrix[c] - self.reference_means[c]) for c in columns)
        out["c_base"] = float(self.stacker.intercept_[0]) + sum(coefficients[c] * self.reference_means[c]
                                                                 for c in self.stack_columns)
        return out

    def escalation(self, scores: pd.DataFrame) -> pd.Series:
        """Jaring pengaman: satu pilar yang sangat yakin sendirian tidak boleh 'tenggelam' di skor gabungan."""
        reasons = pd.Series("", index=scores.index, dtype=object)
        for pillar, column in [("classifier", "s_cls"), ("graph", "s_graph")]:
            limit = self.escalation_thresholds.get(pillar)
            if limit is not None:
                hit = scores[column] >= limit * 100
                reasons[hit] = reasons[hit] + f"{PILLAR_LABELS[pillar]} ≥ {limit * 100:.0f}; "
        return reasons.str.rstrip("; ")

    def risk_band(self, probability, escalation=None) -> np.ndarray:
        probability = np.asarray(probability)
        band = np.select([probability >= self.threshold, probability >= self.watch_threshold], ["Tinggi", "Sedang"],
                         "Rendah")
        if escalation is not None:
            band = np.where((band == "Rendah") & (np.asarray(escalation) != ""), "Sedang", band)
        return band

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        scores = self.pillar_scores(df)
        probability = self.stacker.predict_proba(self.stack_matrix(scores).to_numpy())[:, 1]
        escalation = self.escalation(scores)
        out = pd.DataFrame({"risk_probability": probability, "risk_score": np.round(probability * 100, 1),
                            "risk_band": self.risk_band(probability, escalation), "escalation": escalation.to_numpy()},
                           index=df.index)
        shown = [c for c in ["s_rule", "s_if", "s_cls", "s_graph", "s_graph_evidence"] if c in scores]
        return out.join(scores[shown]).join(self.contributions(df, scores))

    # ------------------------------------------------------------------ bobot untuk dilaporkan
    def learned_weights(self, reference_frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Bobot efektif = |koefisien| × simpangan baku input di data acuan, dinormalisasi jadi 100%."""
        coefficients = dict(zip(self.stack_columns, self.stacker.coef_[0]))
        if reference_frame is not None:
            spread = self.stack_matrix(self.pillar_scores(reference_frame)).std()
        else:
            spread = pd.Series(1.0, index=self.stack_columns)
        rows = []
        for pillar, label in PILLAR_LABELS.items():
            columns = [c for c in self.stack_columns if self.PILLAR_OF_COLUMN[c] == pillar]
            rows.append({"pilar": pillar, "label": label, "koefisien": coefficients[columns[0]],
                         "pengaruh": sum(abs(coefficients[c]) * spread[c] for c in columns)})
        frame = pd.DataFrame(rows)
        frame["bobot_efektif"] = frame["pengaruh"] / frame["pengaruh"].sum()
        return frame

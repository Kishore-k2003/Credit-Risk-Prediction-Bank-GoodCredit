from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sqlalchemy import create_engine


@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


DATE_COLS_ACCOUNT = [
    "dt_opened",
    "upload_dt",
    "opened_dt",
    "last_paymt_dt",
    "closed_dt",
    "reporting_dt",
    "paymt_str_dt",
    "paymt_end_dt",
]
DATE_COLS_ENQUIRY = ["dt_opened", "upload_dt", "enquiry_dt"]
DATE_COLS_DEMO = ["dt_opened", "entry_time"]
NUMERIC_COLS_ACCOUNT = [
    "high_credit_amt",
    "cur_balance_amt",
    "amt_past_due",
    "creditlimit",
    "cashlimit",
    "rateofinterest",
    "actualpaymentamount",
]
NUMERIC_COLS_ENQUIRY = ["enq_amt"]


def _build_uri(cfg: DBConfig) -> str:
    return (
        f"mysql+pymysql://{quote_plus(cfg.user)}:{quote_plus(cfg.password)}"
        f"@{cfg.host}:{cfg.port}/{cfg.database}"
    )


def get_engine(cfg: DBConfig):
    return create_engine(_build_uri(cfg), pool_pre_ping=True)


def load_raw_tables(cfg: DBConfig) -> Dict[str, pd.DataFrame]:
    engine = get_engine(cfg)
    with engine.begin() as conn:
        account = pd.read_sql("SELECT * FROM Cust_Account", conn)
        enquiry = pd.read_sql("SELECT * FROM Cust_Enquiry", conn)
        demo = pd.read_sql("SELECT * FROM Cust_Demographics", conn)
    return {
        "Cust_Account": account,
        "Cust_Enquiry": enquiry,
        "Cust_Demographics": demo,
    }


def _clean_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["customer_no"] = out["customer_no"].astype(str).str.strip()
    return out


def _coerce_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(r"[^0-9\-\.]", "", regex=True)
                .replace({"": np.nan, "nan": np.nan})
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _payment_dpd_features(s: pd.Series) -> pd.DataFrame:
    def parse_dpd(val: str) -> List[int]:
        if pd.isna(val):
            return []
        txt = str(val)
        nums = [int(x) for x in pd.Series(list(txt)).astype(str) if str(x).isdigit()]
        return nums

    rows = []
    for val in s.fillna(""):
        vals = parse_dpd(val)
        if len(vals) == 0:
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        arr = np.array(vals)
        rows.append(
            (
                float(np.mean(arr)),
                float(np.max(arr)),
                float(np.mean((arr >= 3).astype(int))),
                float(len(arr)),
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "paymenthistory_avg_dpd_proxy",
            "paymenthistory_max_dpd_proxy",
            "paymenthistory_30plus_ratio_proxy",
            "paymenthistory_mean_length",
        ],
    )


def preprocess_raw_tables(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    account = _clean_customer_id(raw["Cust_Account"])
    enquiry = _clean_customer_id(raw["Cust_Enquiry"])
    demo = _clean_customer_id(raw["Cust_Demographics"])

    account = _coerce_dates(account, DATE_COLS_ACCOUNT)
    enquiry = _coerce_dates(enquiry, DATE_COLS_ENQUIRY)
    demo = _coerce_dates(demo, DATE_COLS_DEMO)

    account = _coerce_numeric(account, NUMERIC_COLS_ACCOUNT)
    enquiry = _coerce_numeric(enquiry, NUMERIC_COLS_ENQUIRY)

    demo["Bad_label"] = pd.to_numeric(demo["Bad_label"], errors="coerce")

    return {
        "Cust_Account": account,
        "Cust_Enquiry": enquiry,
        "Cust_Demographics": demo,
    }


def engineer_account_features(account: pd.DataFrame) -> pd.DataFrame:
    df = account.copy()

    if {"opened_dt", "last_paymt_dt"}.issubset(df.columns):
        df["diff_lastpaymt_opened_days"] = (df["last_paymt_dt"] - df["opened_dt"]).dt.days
    if {"opened_dt", "reporting_dt"}.issubset(df.columns):
        df["account_age_days"] = (df["reporting_dt"] - df["opened_dt"]).dt.days
    if {"last_paymt_dt", "reporting_dt"}.issubset(df.columns):
        df["days_since_last_payment"] = (df["reporting_dt"] - df["last_paymt_dt"]).dt.days

    payment_proxy_1 = _payment_dpd_features(df.get("paymenthistory1", pd.Series(index=df.index)))
    payment_proxy_2 = _payment_dpd_features(df.get("paymenthistory2", pd.Series(index=df.index)))

    for col in payment_proxy_1.columns:
        df[f"{col}_p1"] = payment_proxy_1[col].values
        df[f"{col}_p2"] = payment_proxy_2[col].values
        df[col] = df[[f"{col}_p1", f"{col}_p2"]].mean(axis=1)

    group = df.groupby("customer_no", as_index=False)
    agg = group.agg(
        account_count=("customer_no", "size"),
        unique_acct_type=("acct_type", "nunique"),
        owner_indic_nunique=("owner_indic", "nunique"),
        high_credit_amt_sum=("high_credit_amt", "sum"),
        high_credit_amt_mean=("high_credit_amt", "mean"),
        cur_balance_amt_sum=("cur_balance_amt", "sum"),
        cur_balance_amt_mean=("cur_balance_amt", "mean"),
        amt_past_due_sum=("amt_past_due", "sum"),
        amt_past_due_mean=("amt_past_due", "mean"),
        creditlimit_sum=("creditlimit", "sum"),
        creditlimit_mean=("creditlimit", "mean"),
        cashlimit_sum=("cashlimit", "sum"),
        cashlimit_mean=("cashlimit", "mean"),
        roi_mean=("rateofinterest", "mean"),
        actualpaymentamount_sum=("actualpaymentamount", "sum"),
        actualpaymentamount_mean=("actualpaymentamount", "mean"),
        diff_lastpaymt_opened_days_mean=("diff_lastpaymt_opened_days", "mean"),
        diff_lastpaymt_opened_days_min=("diff_lastpaymt_opened_days", "min"),
        account_age_days_mean=("account_age_days", "mean"),
        days_since_last_payment_mean=("days_since_last_payment", "mean"),
        paymenthistory_avg_dpd_proxy=("paymenthistory_avg_dpd_proxy", "mean"),
        paymenthistory_max_dpd_proxy=("paymenthistory_max_dpd_proxy", "max"),
        paymenthistory_30plus_ratio_proxy=("paymenthistory_30plus_ratio_proxy", "mean"),
        payment_history_mean_length=("paymenthistory_mean_length", "mean"),
    )

    agg["ratio_currbalance_creditlimit"] = agg["cur_balance_amt_sum"] / agg["creditlimit_sum"].replace(0, np.nan)
    agg["ratio_pastdue_balance"] = agg["amt_past_due_sum"] / agg["cur_balance_amt_sum"].replace(0, np.nan)
    agg["ratio_payment_balance"] = agg["actualpaymentamount_sum"] / agg["cur_balance_amt_sum"].replace(0, np.nan)
    agg["utilisation_trend_proxy"] = (
        (agg["cur_balance_amt_sum"] / agg["creditlimit_sum"].replace(0, np.nan))
        / ((agg["cur_balance_amt_mean"] / (agg["creditlimit_mean"] + agg["cashlimit_mean"]).replace(0, np.nan)).replace(0, np.nan))
    )

    return agg


def engineer_enquiry_features(enquiry: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> pd.DataFrame:
    df = enquiry.copy()
    if ref_date is None:
        ref_date = df["enquiry_dt"].max()

    df["days_since_enquiry"] = (ref_date - df["enquiry_dt"]).dt.days

    group = df.groupby("customer_no", as_index=False)
    agg = group.agg(
        enquiry_count=("customer_no", "size"),
        enq_amt_sum=("enq_amt", "sum"),
        enq_amt_mean=("enq_amt", "mean"),
        enq_amt_max=("enq_amt", "max"),
        unique_enq_purpose=("enq_purpose", "nunique"),
        mean_diff_open_enquiry_dt=("days_since_enquiry", "mean"),
        min_diff_open_enquiry_dt=("days_since_enquiry", "min"),
    )

    recency_90 = df[df["days_since_enquiry"] <= 90].groupby("customer_no").size().rename("count_enquiry_recency_90")
    recency_365 = df[df["days_since_enquiry"] <= 365].groupby("customer_no").size().rename("count_enquiry_recency_365")
    mode_purpose = (
        df.groupby("customer_no")["enq_purpose"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .rename("max_freq_enquiry")
    )

    out = agg.merge(recency_90, on="customer_no", how="left")
    out = out.merge(recency_365, on="customer_no", how="left")
    out = out.merge(mode_purpose, on="customer_no", how="left")

    out["count_enquiry_recency_90"] = out["count_enquiry_recency_90"].fillna(0)
    out["count_enquiry_recency_365"] = out["count_enquiry_recency_365"].fillna(0)
    return out


def build_customer_level_dataset(cleaned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    demo = cleaned["Cust_Demographics"].copy()
    account_feat = engineer_account_features(cleaned["Cust_Account"])
    enquiry_feat = engineer_enquiry_features(cleaned["Cust_Enquiry"])

    data = demo.merge(account_feat, on="customer_no", how="left")
    data = data.merge(enquiry_feat, on="customer_no", how="left")

    return data


def dataset_quality_report(df: pd.DataFrame) -> Dict[str, pd.DataFrame | int | float]:
    missing = (
        df.isna().mean().sort_values(ascending=False).rename("missing_ratio").reset_index().rename(columns={"index": "column"})
    )
    dup_count = int(df.duplicated().sum())
    row_count = int(df.shape[0])
    col_count = int(df.shape[1])

    return {
        "missing": missing,
        "duplicate_count": dup_count,
        "row_count": row_count,
        "col_count": col_count,
    }


def split_features_target(df: pd.DataFrame, target_col: str = "Bad_label") -> Tuple[pd.DataFrame, pd.Series]:
    model_df = df.copy()
    model_df = model_df[model_df[target_col].isin([0, 1])].copy()

    y = model_df[target_col].astype(int)
    X = model_df.drop(columns=[target_col])

    return X, y


def _clip_outliers_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in numeric_cols:
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        out[col] = out[col].clip(low, high)
    return out


def preprocess_for_model(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = X.copy()

    drop_cols = [c for c in ["customer_no", "dt_opened", "entry_time"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object":
            numeric_candidate = pd.to_numeric(df[col], errors="coerce")
            if numeric_candidate.notna().mean() > 0.9:
                df[col] = numeric_candidate

    num_cols = df.select_dtypes(include=[np.number, "datetime64[ns]"]).columns.tolist()
    for col in num_cols:
        if "datetime" in str(df[col].dtype):
            df[col] = df[col].astype("int64") / 10**9

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    df = _clip_outliers_iqr(df, [c for c in num_cols if c in df.columns])

    return df, num_cols, cat_cols


def feature_selection_multi_method(X: pd.DataFrame, y: pd.Series, max_features: int = 60) -> Dict[str, List[str]]:
    # Convert categoricals to frequency-encoded proxy for filter/wrapper methods.
    work = X.copy()
    for col in work.select_dtypes(include=["object"]).columns:
        freq = work[col].value_counts(dropna=False)
        work[col] = work[col].map(freq)

    work = work.fillna(work.median(numeric_only=True))
    sample_n = min(10000, len(work))
    if sample_n < len(work):
        sampled = work.sample(n=sample_n, random_state=42)
        sampled_y = y.loc[sampled.index]
    else:
        sampled = work
        sampled_y = y

    mi = mutual_info_classif(sampled, sampled_y, random_state=42)
    mi_series = pd.Series(mi, index=sampled.columns).sort_values(ascending=False)
    mi_top = mi_series.head(min(max_features, len(mi_series))).index.tolist()

    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced_subsample", max_depth=8)
    rf.fit(sampled, sampled_y)
    rf_imp = pd.Series(rf.feature_importances_, index=sampled.columns).sort_values(ascending=False)
    rf_top = rf_imp.head(min(max_features, len(rf_imp))).index.tolist()

    n_select = min(20, max(5, int(math.sqrt(sampled.shape[1]))))
    lr_rfe = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")
    selector = RFE(lr_rfe, n_features_to_select=n_select)
    selector.fit(sampled.fillna(0), sampled_y)
    rfe_top = sampled.columns[selector.support_].tolist()

    selected = sorted(list(set(mi_top).intersection(set(rf_top)).union(set(rfe_top))))

    return {
        "mutual_info_top": mi_top,
        "random_forest_top": rf_top,
        "rfe_top": rfe_top,
        "selected": selected,
    }


def _build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric: bool = False) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def ks_statistic(y_true: pd.Series, y_prob: np.ndarray) -> float:
    bad = y_prob[y_true == 1]
    good = y_prob[y_true == 0]
    if len(bad) == 0 or len(good) == 0:
        return np.nan
    return float(ks_2samp(bad, good).statistic)


def decile_table(y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=n_bins, labels=False) + 1

    tab = (
        df.groupby("decile", as_index=False)
        .agg(
            customers=("y_true", "count"),
            bads=("y_true", "sum"),
            min_prob=("y_prob", "min"),
            max_prob=("y_prob", "max"),
            avg_prob=("y_prob", "mean"),
        )
        .sort_values("decile")
    )
    tab["goods"] = tab["customers"] - tab["bads"]
    tab["bad_rate"] = tab["bads"] / tab["customers"]
    tab["cum_bads"] = tab["bads"].cumsum()
    tab["cum_goods"] = tab["goods"].cumsum()
    tab["cum_bad_pct"] = tab["cum_bads"] / tab["bads"].sum().clip(lower=1)
    tab["cum_customer_pct"] = tab["customers"].cumsum() / tab["customers"].sum().clip(lower=1)
    tab["lift"] = tab["bad_rate"] / (tab["bads"].sum() / tab["customers"].sum())
    return tab


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    auc = roc_auc_score(y_true, y_prob)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_prob),
    }


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, selected_features: List[str]) -> Dict[str, object]:
    use_cols = [c for c in selected_features if c in X.columns]
    X_use = X[use_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_use, y, test_size=0.25, random_state=42, stratify=y
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    models = {}

    # Baseline logistic regression with class weighting.
    prep_lr = _build_preprocessor(num_cols, cat_cols, scale_numeric=True)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe_lr = Pipeline([("prep", prep_lr), ("model", lr)])
    pipe_lr.fit(X_train, y_train)
    models["logistic_class_weight"] = pipe_lr

    # SMOTE + logistic regression for imbalance treatment comparison.
    smote_lr = LogisticRegression(max_iter=2000)
    pipe_smote_lr = ImbPipeline([
        ("prep", prep_lr),
        ("smote", SMOTE(random_state=42)),
        ("model", smote_lr),
    ])
    pipe_smote_lr.fit(X_train, y_train)
    models["logistic_smote"] = pipe_smote_lr

    # Random forest baseline ensemble.
    prep_rf = _build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    pipe_rf = Pipeline([("prep", prep_rf), ("model", rf)])
    pipe_rf.fit(X_train, y_train)
    models["random_forest"] = pipe_rf

    # Optional gradient boosting ensemble with tuning.
    if XGB_AVAILABLE:
        prep_xgb = _build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        pipe_xgb = Pipeline([("prep", prep_xgb), ("model", xgb)])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        param_dist = {
            "model__n_estimators": [250, 350, 450],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
        }

        search = RandomizedSearchCV(
            pipe_xgb,
            param_distributions=param_dist,
            n_iter=5,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_train, y_train)
        models["xgboost_tuned"] = search.best_estimator_

    results_rows = []
    eval_details = {}

    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        deciles = decile_table(y_test, y_prob, n_bins=10)
        cm = confusion_matrix(y_test, y_pred)
        pr_precision, pr_recall, pr_threshold = precision_recall_curve(y_test, y_prob)

        out = {"model": model_name}
        out.update(metrics)
        results_rows.append(out)

        eval_details[model_name] = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "decile_table": deciles,
            "y_test": y_test,
            "y_prob": y_prob,
            "pr_precision": pr_precision,
            "pr_recall": pr_recall,
            "pr_threshold": pr_threshold,
            "model": model,
        }

    results_df = pd.DataFrame(results_rows).sort_values(["gini", "ks", "recall"], ascending=False)
    best_model_name = results_df.iloc[0]["model"]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "results_df": results_df,
        "eval_details": eval_details,
        "best_model_name": best_model_name,
    }


def feature_importance_from_pipeline(model: Pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    est = model.named_steps["model"]

    feature_names = prep.get_feature_names_out()

    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

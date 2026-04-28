import io
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import DMatrix, XGBClassifier


st.set_page_config(
    page_title="Intelligent Loan Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    # UI Improvement added
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f4f8ff 0%, #f8fbff 100%);
        }
        .hero-card {
            padding: 1.2rem 1.4rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .section-card {
            background: #ffffff;
            padding: 1rem 1.1rem;
            border-radius: 14px;
            box-shadow: 0 6px 18px rgba(21, 61, 130, 0.10);
            border: 1px solid #edf2ff;
        }
        .metric-card {
            background: linear-gradient(145deg, #ffffff 0%, #f7faff 100%);
            padding: 0.9rem;
            border-radius: 12px;
            border: 1px solid #e9f0ff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            text-align: center;
        }
        .approved {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f9d58;
            animation: pulse 1.6s infinite;
        }
        .rejected {
            font-size: 1.6rem;
            font-weight: 700;
            color: #d93025;
            animation: pulse 1.6s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.02); opacity: 0.92; }
            100% { transform: scale(1); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    return train_df, test_df


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for df in (train_df, test_df):
        if "Loan_ID" in df.columns:
            df.drop(columns=["Loan_ID"], inplace=True)

    target_col = "Loan_Status"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols].copy()
    y_train_raw = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test_raw = test_df[target_col].copy() if target_col in test_df.columns else None

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if num_cols:
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    if cat_cols:
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    feature_encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat(
            [X_train[col].astype(str), X_test[col].astype(str)],
            axis=0,
        )
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        feature_encoders[col] = le

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train_raw.astype(str))
    y_test = (
        target_encoder.transform(y_test_raw.astype(str))
        if y_test_raw is not None
        else None
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_encoders": feature_encoders,
        "target_encoder": target_encoder,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "train_df": train_df,
    }


@st.cache_resource
def train_models(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> Dict[str, Any]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_split=8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(
    models: Dict[str, Any], X_test: pd.DataFrame, y_test: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    evaluation: Dict[str, Dict[str, Any]] = {}
    for model_name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        evaluation[model_name] = {
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
            "y_pred": preds,
            "y_proba": proba,
        }
    return evaluation


def predict(
    input_df: pd.DataFrame,
    pipeline: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    x = input_df.copy()

    if pipeline["num_cols"]:
        x[pipeline["num_cols"]] = pipeline["num_imputer"].transform(x[pipeline["num_cols"]])
    if pipeline["cat_cols"]:
        x[pipeline["cat_cols"]] = pipeline["cat_imputer"].transform(x[pipeline["cat_cols"]])

    for col, le in pipeline["feature_encoders"].items():
        val = x[col].astype(str).iloc[0]
        if val not in set(le.classes_):
            val = le.classes_[0]
        x[col] = le.transform([val])

    probability = float(model.predict_proba(x)[:, 1][0])
    pred_num = int(model.predict(x)[0])
    pred_label = pipeline["target_encoder"].inverse_transform([pred_num])[0]

    explanation_df = pd.DataFrame(columns=["Feature", "Contribution"])
    if isinstance(model, XGBClassifier):
        booster = model.get_booster()
        dmat = DMatrix(x, feature_names=list(x.columns))
        contribs = booster.predict(dmat, pred_contribs=True)[0]
        feat_contribs = pd.DataFrame(
            {
                "Feature": list(x.columns) + ["Bias"],
                "Contribution": contribs,
            }
        )
        explanation_df = (
            feat_contribs[feat_contribs["Feature"] != "Bias"]
            .reindex(feat_contribs["Contribution"].abs().sort_values(ascending=False).index)
            .head(6)
            .reset_index(drop=True)
        )

    return {
        "prediction_label": pred_label,
        "approval_probability": probability,
        "encoded_input": x,
        "explanation_df": explanation_df,
    }


def plot_roc_curves(evaluation: Dict[str, Dict[str, Any]], y_test: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for model_name, metrics in evaluation.items():
        fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={metrics['roc_auc']:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Baseline",
        )
    )
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=430,
    )
    return fig


def main() -> None:
    inject_css()
    train_df, test_df = load_data()
    pipeline = preprocess(train_df, test_df)
    models = train_models(pipeline["X_train"], pipeline["y_train"])

    has_test_target = pipeline["y_test"] is not None
    if has_test_target:
        eval_models = models
        eval_X = pipeline["X_test"]
        eval_y = pipeline["y_test"]
        eval_data_note = "Evaluation source: labeled `test.csv`."
    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            pipeline["X_train"],
            pipeline["y_train"],
            test_size=0.2,
            random_state=42,
            stratify=pipeline["y_train"],
        )
        eval_models = train_models(X_fit, y_fit)
        eval_X = X_val
        eval_y = y_val
        eval_data_note = (
            "Evaluation source: fallback validation split from `train.csv` "
            "(because `test.csv` has no `Loan_Status`)."
        )

    evaluation = evaluate_models(eval_models, eval_X, eval_y)

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.2rem;">Intelligent Loan Approval Prediction System</h1>
            <p style="margin:0; font-size:1rem;">
                Production-grade machine learning dashboard for loan decision intelligence, risk profiling, and explainable prediction insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Smart Application Input")
        st.caption("Tune values using intuitive controls and get instant approval insights.")

        with st.expander("Personal Info", expanded=True):
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])

        with st.expander("Financial Info", expanded=True):
            applicant_income = st.slider("Applicant Income", 0, 100000, 5000, 100)
            coapplicant_income = st.slider("Coapplicant Income", 0, 60000, 1500, 100)
            credit_history = st.selectbox("Credit History", [1.0, 0.0])

        with st.expander("Loan Details", expanded=True):
            loan_amount = st.slider("Loan Amount (in thousands)", 1, 1000, 130, 1)
            loan_amount_term = st.slider("Loan Amount Term (months)", 12, 480, 360, 12)
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        model_choice = st.selectbox("Model for Prediction", list(models.keys()), index=2)
        show_advanced = st.toggle("Show Advanced Analytics", value=True)
        predict_btn = st.button("Predict Loan Approval", use_container_width=True)

    input_data = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }
        ]
    )

    if show_advanced:
        st.subheader("Data Insights Panel")
        insight_col1, insight_col2 = st.columns([1, 1.2])
        with insight_col1:
            class_counts = pipeline["train_df"]["Loan_Status"].value_counts()
            pie_fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Approved vs Rejected Distribution",
                color=class_counts.index,
                color_discrete_map={"Y": "#12b76a", "N": "#f04438"},
            )
            pie_fig.update_traces(textposition="inside", textinfo="percent+label")
            pie_fig.update_layout(height=360, template="plotly_white")
            st.plotly_chart(pie_fig, use_container_width=True)
        with insight_col2:
            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"<div class='metric-card'><h4>Train Rows</h4><h3>{pipeline['X_train'].shape[0]}</h3></div>",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"<div class='metric-card'><h4>Features</h4><h3>{pipeline['X_train'].shape[1]}</h3></div>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"<div class='metric-card'><h4>Test Rows</h4><h3>{pipeline['X_test'].shape[0]}</h3></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            desc_df = pipeline["train_df"].describe(include="all").T.reset_index().rename(columns={"index": "Column"})
            st.dataframe(desc_df, use_container_width=True, height=250)

    st.subheader("Model Performance Panel")
    st.caption(eval_data_note)
    eval_df = pd.DataFrame(
        {
            "Model": list(evaluation.keys()),
            "Accuracy": [v["accuracy"] for v in evaluation.values()],
            "ROC AUC": [v["roc_auc"] for v in evaluation.values()],
        }
    ).sort_values(by="Accuracy", ascending=False)
    winner = eval_df.iloc[0]["Model"]
    winner_suffix = "test set" if has_test_target else "validation split"
    st.success(f"🏆 Model Comparison Winner: **{winner}** (highest accuracy on {winner_suffix})")

    perf_col1, perf_col2 = st.columns([1, 1])
    with perf_col1:
        bar_fig = px.bar(
            eval_df,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Model Accuracy Comparison",
            text=eval_df["Accuracy"].map(lambda x: f"{x:.3f}"),
        )
        bar_fig.update_layout(template="plotly_white", height=360, showlegend=False)
        st.plotly_chart(bar_fig, use_container_width=True)

    with perf_col2:
        st.plotly_chart(plot_roc_curves(evaluation, eval_y), use_container_width=True)

    matrix_col1, matrix_col2 = st.columns([1, 1])
    with matrix_col1:
        cm = confusion_matrix(eval_y, evaluation[winner]["y_pred"])
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Rejected", "Approved"],
            y=["Rejected", "Approved"],
            title=f"Confusion Matrix ({winner})",
            color_continuous_scale="Blues",
        )
        cm_fig.update_layout(height=360)
        st.plotly_chart(cm_fig, use_container_width=True)

    with matrix_col2:
        xgb_model = eval_models["XGBoost"]
        importances = pd.DataFrame(
            {
                "Feature": pipeline["X_train"].columns,
                "Importance": xgb_model.feature_importances_,
            }
        ).sort_values(by="Importance", ascending=False).head(12)
        imp_fig = px.bar(
            importances[::-1],
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            title="Top XGBoost Feature Importance",
        )
        imp_fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(imp_fig, use_container_width=True)

    st.subheader("Prediction Panel")
    if predict_btn:
        with st.spinner("Analyzing applicant profile and generating risk intelligence..."):
            selected_model = models[model_choice]
            result = predict(input_data, pipeline, selected_model)

        is_approved = result["prediction_label"] == "Y"
        result_text = "Loan Approved ✅" if is_approved else "Loan Rejected ❌"
        result_class = "approved" if is_approved else "rejected"

        pred_col1, pred_col2 = st.columns([1, 1.1])
        with pred_col1:
            st.markdown(
                f"""
                <div class="section-card">
                    <p style="margin:0 0 0.4rem 0; color:#667085;">Prediction Result</p>
                    <div class="{result_class}">{result_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            risk_prob = result["approval_probability"] * 100
            if risk_prob >= 70:
                risk_label, risk_color = "Low Risk", "green"
            elif risk_prob >= 45:
                risk_label, risk_color = "Medium Risk", "#f59e0b"
            else:
                risk_label, risk_color = "High Risk", "red"

            st.markdown(
                f"<div class='section-card'><b>Credit Risk Indicator:</b> <span style='color:{risk_color}'>{risk_label}</span></div>",
                unsafe_allow_html=True,
            )

        with pred_col2:
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_prob,
                    title={"text": "Approval Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#5b8def"},
                        "steps": [
                            {"range": [0, 45], "color": "#ffe0e0"},
                            {"range": [45, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#d9fbe6"},
                        ],
                    },
                )
            )
            gauge_fig.update_layout(height=260, template="plotly_white")
            st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown("#### Explanation Panel")
        if not result["explanation_df"].empty:
            exp_fig = px.bar(
                result["explanation_df"][::-1],
                x="Contribution",
                y="Feature",
                orientation="h",
                color="Contribution",
                color_continuous_scale="RdYlGn",
                title="Top Features Influencing This Prediction (XGBoost SHAP-like contributions)",
            )
            exp_fig.update_layout(template="plotly_white", height=360)
            st.plotly_chart(exp_fig, use_container_width=True)
        else:
            st.info("Feature-level explanation is shown when `XGBoost` is selected.")

        export_df = pd.DataFrame(
            [
                {
                    "Model": model_choice,
                    "Prediction": result_text,
                    "Approval_Probability_Percent": round(risk_prob, 2),
                    "Credit_Risk_Level": risk_label,
                }
            ]
        )
        buffer = io.StringIO()
        export_df.to_csv(buffer, index=False)
        st.download_button(
            "Download Prediction as CSV",
            data=buffer.getvalue(),
            file_name="loan_prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Fill applicant details in the sidebar and click **Predict Loan Approval**.")


if __name__ == "__main__":
    main()
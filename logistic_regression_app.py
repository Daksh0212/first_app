import streamlit as st
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------
# Streamlit app: Logistic Regression on the Iris dataset
# ---------------------------------------------------------

st.set_page_config(
    page_title="Logistic Regression Playground",
    page_icon="🌸",
    layout="wide",
)

st.title("🌸 Logistic Regression Playground")
st.write(
    "Explore a basic binary logistic regression classifier using the classic "
    "Iris dataset. Choose two flower classes, adjust the measurements, and "
    "see the model's prediction."
)

iris = load_iris(as_frame=True)
X_all = iris.data
y_all = iris.target
target_names = list(iris.target_names)
feature_names = list(X_all.columns)

st.sidebar.header("Model settings")

class_a = st.sidebar.selectbox(
    "First class",
    target_names,
    index=0,
)
class_b_options = [name for name in target_names if name != class_a]
class_b = st.sidebar.selectbox(
    "Second class",
    class_b_options,
    index=0,
)

test_size = st.sidebar.slider(
    "Test set size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

random_state = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10000,
    value=42,
    step=1,
)

# Keep only the two selected classes and encode them as 0 and 1.
class_a_id = target_names.index(class_a)
class_b_id = target_names.index(class_b)
selected_mask = y_all.isin([class_a_id, class_b_id])

X = X_all.loc[selected_mask].copy()
y_original = y_all.loc[selected_mask]
y = (y_original == class_b_id).astype(int)

# Split first, then fit preprocessing and model only on the training data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=int(random_state),
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=int(random_state)),
)
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, test_predictions)
cm = confusion_matrix(y_test, test_predictions, labels=[0, 1])

# Main layout
overview_col, metrics_col = st.columns([1.4, 1])

with overview_col:
    st.subheader("1. Dataset")
    st.write(
        f"The app is currently distinguishing **{class_a}** (class 0) "
        f"from **{class_b}** (class 1)."
    )

    dataset_summary = pd.DataFrame(
        {
            "Class": [class_a, class_b],
            "Samples": [
                int((y == 0).sum()),
                int((y == 1).sum()),
            ],
        }
    )
    st.dataframe(dataset_summary, hide_index=True, use_container_width=True)

with metrics_col:
    st.subheader("2. Model performance")
    st.metric("Test accuracy", f"{accuracy:.1%}")
    st.caption(
        f"Trained on {len(X_train)} samples and evaluated on {len(X_test)} "
        "held-out samples."
    )

st.divider()
st.subheader("3. Try your own flower measurements")
st.write(
    "Change the measurements below. The model will estimate which of the "
    "two selected classes is more likely."
)

input_columns = st.columns(2)
user_values = {}

for i, feature in enumerate(feature_names):
    minimum = float(X_all[feature].min())
    maximum = float(X_all[feature].max())
    default = float(X_all[feature].median())

    with input_columns[i % 2]:
        user_values[feature] = st.slider(
            feature.replace(" (cm)", "").title() + " (cm)",
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=0.1,
        )

input_df = pd.DataFrame([user_values], columns=feature_names)
predicted_class = int(model.predict(input_df)[0])
probabilities = model.predict_proba(input_df)[0]

probability_a = float(probabilities[0])
probability_b = float(probabilities[1])

result_col, probability_col = st.columns([1, 1])

with result_col:
    st.subheader("Prediction")
    predicted_name = class_b if predicted_class == 1 else class_a
    st.success(f"Predicted class: **{predicted_name}**")

with probability_col:
    st.subheader("Predicted probabilities")
    probability_df = pd.DataFrame(
        {
            "Class": [class_a, class_b],
            "Probability": [probability_a, probability_b],
        }
    )
    st.bar_chart(probability_df.set_index("Class"))

st.divider()
st.subheader("4. Evaluation details")

with st.expander("Confusion matrix"):
    st.write("Rows are the actual classes; columns are the predicted classes.")
    confusion_df = pd.DataFrame(
        cm,
        index=[f"Actual {class_a}", f"Actual {class_b}"],
        columns=[f"Predicted {class_a}", f"Predicted {class_b}"],
    )
    st.dataframe(confusion_df, use_container_width=True)

with st.expander("Classification report"):
    report = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=[class_a, class_b],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

with st.expander("How this app works"):
    st.markdown(
        """
        1. It loads the built-in Iris dataset from scikit-learn.
        2. It keeps only the two flower classes selected in the sidebar.
        3. It splits the data into training and test sets.
        4. `StandardScaler` standardizes the four measurements.
        5. `LogisticRegression` learns a binary classification boundary.
        6. The trained pipeline predicts the class and probabilities for the
           measurements you enter.

        **Why scale the features?** The four measurements can have different
        numerical ranges. Standardization helps logistic regression train
        consistently and makes its regularization behave sensibly.
        """
    )

st.caption(
    "Educational demo only. The Iris dataset is a small, clean teaching "
    "dataset; results on real-world data may be less straightforward."
)

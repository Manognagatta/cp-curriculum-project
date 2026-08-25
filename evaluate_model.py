import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from data_generator import CPDatasetGenerator
from model import CompetitiveProgrammingModel
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

print("Loading dataset...")
generator = CPDatasetGenerator(num_problems=500)
solutions_df, problems, students = generator.save_dataset()

print("Preparing data...")

# Updated 9 input features
feature_cols = [
    'lines_of_code',
    'cyclomatic_complexity',
    'num_functions',
    'has_recursion',
    'num_loops',
    'execution_time',
    'memory_used_mb',
    'time_to_solve_minutes',
    'algorithmic_complexity'
]

X = solutions_df[feature_cols].values

# Difficulty target
difficulty_map = {
    "Easy": [1, 0, 0],
    "Medium": [0, 1, 0],
    "Hard": [0, 0, 1]
}

y_difficulty = np.array([
    difficulty_map[d]
    for d in solutions_df['difficulty']
])

# Skill target
y_skill = (
    1 - (
        solutions_df['execution_time']
        / solutions_df['execution_time'].max()
    )
) * 100

y_skill = y_skill.values.reshape(-1, 1) / 100

# Topic target
topics = [
    "Array",
    "String",
    "DP",
    "Graph",
    "Tree",
    "Greedy",
    "Math",
    "Heap",
    "HashTable",
    "Stack"
]

topic_map = {
    topic: np.eye(10)[i]
    for i, topic in enumerate(topics)
}

y_topic = np.array([
    topic_map[t]
    for t in solutions_df['topic']
])

# Success target
y_success = solutions_df['passes_all_tests'].values.reshape(-1, 1)

# Same split used during training
from sklearn.model_selection import GroupShuffleSplit

groups = solutions_df["problem_id"].values

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, val_idx = next(
    gss.split(X, y_difficulty, groups=groups)
)

X_train = X[train_idx]
X_val = X[val_idx]

y_diff_train = y_difficulty[train_idx]
y_diff_val = y_difficulty[val_idx]

y_skill_train = y_skill[train_idx]
y_skill_val = y_skill[val_idx]

y_topic_train = y_topic[train_idx]
y_topic_val = y_topic[val_idx]

y_success_train = y_success[train_idx]
y_success_val = y_success[val_idx]

print("Loading trained model...")

model_wrapper = CompetitiveProgrammingModel(
    input_features=len(feature_cols)
)

model_wrapper.build_model()
model_wrapper.load(model_dir="models")

print("\n" + "=" * 60)
print("MODEL ACCURACY EVALUATION")
print("=" * 60)

# Make predictions
predictions = model_wrapper.predict(X_val)

# ============================================================
# TASK 1: DIFFICULTY CLASSIFICATION
# ============================================================

y_diff_pred = np.argmax(
    predictions['difficulty'],
    axis=1
)

y_diff_true = np.argmax(
    y_diff_val,
    axis=1
)

difficulty_accuracy = accuracy_score(
    y_diff_true,
    y_diff_pred
)

print("\n📊 TASK 1: Difficulty Classification (Easy/Medium/Hard)")
print(f"   Accuracy: {difficulty_accuracy * 100:.1f}%")

# ============================================================
# TASK 2: SKILL LEVEL REGRESSION
# ============================================================

skill_mae = np.mean(
    np.abs(
        predictions['skill']
        - (y_skill_val.flatten() * 100)
    )
)

skill_rmse = np.sqrt(
    mean_squared_error(
        y_skill_val.flatten() * 100,
        predictions['skill']
    )
)

print("\n📊 TASK 2: Skill Level Regression (0-100)")
print(f"   Mean Absolute Error: {skill_mae:.2f} points")
print(f"   RMSE: {skill_rmse:.2f}")

# ============================================================
# TASK 3: TOPIC CLASSIFICATION
# ============================================================

y_topic_pred = np.argmax(
    predictions['topic'],
    axis=1
)

y_topic_true = np.argmax(
    y_topic_val,
    axis=1
)

topic_accuracy = accuracy_score(
    y_topic_true,
    y_topic_pred
)

print("\n📊 TASK 3: Topic Classification (10 topics)")
print(f"   Accuracy: {topic_accuracy * 100:.1f}%")

# ============================================================
# TASK 4: SUCCESS PREDICTION
# ============================================================

y_success_pred = (
    predictions['success'] > 0.5
).astype(int)

y_success_true = (
    y_success_val.flatten()
    .astype(int)
)

success_accuracy = accuracy_score(
    y_success_true,
    y_success_pred
)

success_f1 = f1_score(
    y_success_true,
    y_success_pred
)

print("\n📊 TASK 4: Success Prediction (Will they solve?)")
print(f"   Accuracy: {success_accuracy * 100:.1f}%")
print(f"   F1-Score: {success_f1:.3f}")

# ============================================================
# OVERALL ACCURACY
# ============================================================

overall_accuracy = (
    difficulty_accuracy
    + topic_accuracy
    + success_accuracy
) / 3

print("\n" + "=" * 60)
print(
    f"OVERALL MULTI-TASK ACCURACY: "
    f"{overall_accuracy * 100:.1f}%"
)
print("=" * 60)

# ============================================================
# SUMMARY
# ============================================================

print("\n✅ Model Performance Summary:")
print(
    f"   - Difficulty prediction: "
    f"{difficulty_accuracy * 100:.1f}% ✓"
)

print(
    f"   - Skill estimation error: "
    f"{skill_mae:.1f} points"
)

print(
    f"   - Topic prediction: "
    f"{topic_accuracy * 100:.1f}% ✓"
)

print(
    f"   - Success prediction: "
    f"{success_accuracy * 100:.1f}% ✓"
)
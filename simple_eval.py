import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_generator import CPDatasetGenerator
from model import CompetitiveProgrammingModel

print("Loading dataset...")
generator = CPDatasetGenerator(num_problems=500)
solutions_df, problems, students = generator.save_dataset()

print("Preparing data...")
feature_cols = ['lines_of_code', 'cyclomatic_complexity', 'num_functions', 'has_recursion', 'num_loops', 'execution_time', 'memory_used_mb', 'time_to_solve_minutes', 'algorithmic_complexity']
X = solutions_df[feature_cols].values

y_difficulty = np.array([[1,0,0] if d == "Easy" else [0,1,0] if d == "Medium" else [0,0,1] for d in solutions_df['difficulty']])
y_skill = ((1 - solutions_df['execution_time'] / solutions_df['execution_time'].max()) * 100).values.reshape(-1, 1) / 100
topics = ["Array", "String", "DP", "Graph", "Tree", "Greedy", "Math", "Heap", "HashTable", "Stack"]
y_topic = np.array([[1 if t == topic else 0 for topic in topics] for t in solutions_df['topic']])
y_success = solutions_df['passes_all_tests'].values.reshape(-1, 1)

X_train, X_val, y_diff_train, y_diff_val, y_skill_train, y_skill_val, y_topic_train, y_topic_val, y_success_train, y_success_val = train_test_split(
    X, y_difficulty, y_skill, y_topic, y_success, test_size=0.2, random_state=42
)

print("Building and training model...")
model = CompetitiveProgrammingModel(len(feature_cols))
model.build_model()
train_targets = {'difficulty': y_diff_train, 'skill': y_skill_train, 'topic': y_topic_train, 'success': y_success_train}
val_targets = {'difficulty': y_diff_val, 'skill': y_skill_val, 'topic': y_topic_val, 'success': y_success_val}
model.train(X_train, train_targets, X_val, val_targets, epochs=50)

print("\n" + "="*60)
print("MODEL ACCURACY RESULTS")
print("="*60)

predictions = model.predict(X_val)

y_diff_pred = np.argmax(predictions['difficulty'], axis=1)
y_diff_true = np.argmax(y_diff_val, axis=1)
diff_acc = accuracy_score(y_diff_true, y_diff_pred) * 100

y_topic_pred = np.argmax(predictions['topic'], axis=1)
y_topic_true = np.argmax(y_topic_val, axis=1)
topic_acc = accuracy_score(y_topic_true, y_topic_pred) * 100

y_success_pred = (predictions['success'] > 0.5).astype(int)
y_success_true = y_success_val.flatten().astype(int)
success_acc = accuracy_score(y_success_true, y_success_pred) * 100

skill_error = np.mean(np.abs(predictions['skill'] - (y_skill_val.flatten() * 100)))

print(f"\n📊 Task 1: Difficulty Classification")
print(f"   Accuracy: {diff_acc:.1f}%")

print(f"\n📊 Task 2: Skill Level Estimation")
print(f"   MAE: {skill_error:.1f} points")

print(f"\n📊 Task 3: Topic Classification")
print(f"   Accuracy: {topic_acc:.1f}%")

print(f"\n📊 Task 4: Success Prediction")
print(f"   Accuracy: {success_acc:.1f}%")

overall = (diff_acc + topic_acc + success_acc) / 3
print(f"\n" + "="*60)
print(f"OVERALL ACCURACY: {overall:.1f}%")
print("="*60)

print(f"\n✅ Model Performance:")
print(f"   - Difficulty: {diff_acc:.1f}% ✓")
print(f"   - Skill Error: ±{skill_error:.1f} points")
print(f"   - Topic: {topic_acc:.1f}% ✓")
print(f"   - Success: {success_acc:.1f}% ✓")

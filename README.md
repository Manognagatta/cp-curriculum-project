# Adaptive Competitive Programming Curriculum

A multi-task deep learning system for adaptive competitive programming practice.

The project uses code-level and problem-level features to jointly predict:

1. **Problem Difficulty** — Easy / Medium / Hard
2. **Student Skill Level** — estimated on a 0–100 scale
3. **Problem Topic** — 10 competitive-programming topics
4. **Success Probability** — probability that a student will solve the problem

The goal is to support an adaptive curriculum that can use these predictions to select problems that better match a student's current level and learning needs.

## Model Architecture

The core model is a multi-task neural network with a shared representation and task-specific prediction branches.

```text
Input Features (9)
        |
        v
Shared Dense Layers
   128 -> 64
        |
   +----+----+----------------+
   |         |                |
Difficulty  Skill          Success
   |         |                |
   +---------+----------------+
             |
       Topic-specific
          Encoder
             |
           Topic
```

### Input Features

The model uses these nine features:

- `lines_of_code`
- `cyclomatic_complexity`
- `num_functions`
- `has_recursion`
- `num_loops`
- `execution_time`
- `memory_used_mb`
- `time_to_solve_minutes`
- `algorithmic_complexity`

The model standardizes the input features before training and prediction.

## Dataset

The project includes a synthetic competitive-programming dataset generator.

The generator creates:

- 500 programming problems
- 10 topics
- 3 difficulty levels
- 5 generated solutions per problem
- student submission history

Topics:

```text
Array
String
DP
Graph
Tree
Greedy
Math
Heap
HashTable
Stack
```

The training pipeline uses a **problem-level grouped train/validation split**, keeping all five solutions belonging to the same problem in the same split. This prevents correlated solutions for one problem from appearing in both training and validation.

## Performance

Final evaluation results from the current model:

| Task | Result |
|---|---:|
| Difficulty Classification | **99.8% accuracy** |
| Skill Level Regression | **2.6 point MAE** |
| Topic Classification | **54.8% accuracy** |
| Success Prediction | **68.8% accuracy** |
| Success Prediction | **0.758 F1-score** |

The project also reports a **74.5% multi-task accuracy summary** across the three classification tasks (difficulty, topic, and success). Skill estimation is a regression task and is therefore not included in that aggregate.

## Project Structure

```text
cp-curriculum-project/
├── app.py
├── curriculum_engine.py
├── data_generator.py
├── evaluate_model.py
├── model.py
├── simple_eval.py
├── requirements.txt
├── .gitignore
└── CP Curriculum Presentation 23slides with Comparison.pptx
```

Generated runtime files are intentionally excluded from Git with `.gitignore`.

## Installation

Clone the repository:

```bash
git clone https://github.com/Manognagatta/cp-curriculum-project.git
cd cp-curriculum-project
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset

```bash
python3 data_generator.py
```

This creates the generated dataset under:

```text
data/
```

### 2. Train the model

```bash
python3 model.py
```

The trained model and scaler are saved under:

```text
models/
```

### 3. Evaluate the trained model

```bash
python3 evaluate_model.py
```

This reports the performance of the four prediction tasks.

## Key Design Choices

### Multi-task learning

Difficulty, skill, topic, and success are learned together using a shared representation. This allows the model to learn common patterns while maintaining separate task-specific prediction heads.

### Topic-specific representation

Topic classification uses a dedicated encoder from the normalized input features. This gives the topic task additional capacity because topic information is distributed across several noisy code characteristics.

### Leakage-aware validation

Because each problem has multiple generated solutions, validation is split using `problem_id` groups rather than individual solution rows. This prevents solutions belonging to the same underlying problem from crossing the train/validation boundary.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Joblib

## Future Work

- Integrate the predictions into a fully adaptive problem recommendation loop
- Incorporate richer student history and performance features
- Add explainability for recommendations and predictions
- Evaluate on real competitive-programming submission data
- Compare against dedicated single-task baselines

## Repository

GitHub:

https://github.com/Manognagatta/cp-curriculum-project

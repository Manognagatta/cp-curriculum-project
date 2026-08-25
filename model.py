"""
Multi-Task Deep Learning Model for Competitive Programming
Predicts: difficulty, skill level, topic, and success probability
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os


class CompetitiveProgrammingModel:
    """
    Multi-task learning model using shared embeddings.

    Tasks:
    1. Difficulty Classification (Easy/Medium/Hard)
    2. Skill Level Regression (0-100)
    3. Topic Classification (10 topics)
    4. Success Prediction (will solve or not)
    """

    def __init__(self, input_features: int = 9, num_topics: int = 10):
        self.input_features = input_features
        self.num_topics = num_topics
        self.model = None
        self.scaler = StandardScaler()

        self.feature_names = [
            "lines_of_code",
            "cyclomatic_complexity",
            "num_functions",
            "has_recursion",
            "num_loops",
            "execution_time",
            "memory_used_mb",
            "time_to_solve_minutes",
            "algorithmic_complexity",
        ]

    def build_model(self):
        """
        Build the multi-task neural network.

        Important design choice:
        The topic branch receives the normalized input directly instead of
        relying only on the shared representation. This prevents the
        difficulty/success tasks from dominating the representation used
        for topic classification.
        """

        inputs = keras.Input(
            shape=(self.input_features,),
            name="input"
        )

        # ============================================================
        # SHARED REPRESENTATION
        # ============================================================
        shared = layers.Dense(
            128,
            activation="relu",
            name="shared_dense_1"
        )(inputs)

        shared = layers.BatchNormalization(
            name="shared_bn_1"
        )(shared)

        shared = layers.Dropout(
            0.25,
            name="shared_dropout_1"
        )(shared)

        shared = layers.Dense(
            64,
            activation="relu",
            name="shared_dense_2"
        )(shared)

        shared = layers.BatchNormalization(
            name="shared_bn_2"
        )(shared)

        shared = layers.Dropout(
            0.20,
            name="shared_dropout_2"
        )(shared)

        # ============================================================
        # TASK 1: DIFFICULTY CLASSIFICATION
        # ============================================================
        difficulty_branch = layers.Dense(
            32,
            activation="relu",
            name="difficulty_dense_1"
        )(shared)

        difficulty_branch = layers.Dropout(
            0.15,
            name="difficulty_dropout"
        )(difficulty_branch)

        difficulty_out = layers.Dense(
            3,
            activation="softmax",
            name="difficulty"
        )(difficulty_branch)

        # ============================================================
        # TASK 2: SKILL LEVEL REGRESSION
        # ============================================================
        skill_branch = layers.Dense(
            32,
            activation="relu",
            name="skill_dense_1"
        )(shared)

        skill_branch = layers.Dropout(
            0.15,
            name="skill_dropout"
        )(skill_branch)

        skill_out = layers.Dense(
            1,
            activation="sigmoid",
            name="skill"
        )(skill_branch)

        # ============================================================
        # TASK 3: TOPIC CLASSIFICATION
        # ============================================================
        # DO NOT make this branch depend only on `shared`.
        # The generator encodes topic information in:
        # complexity, functions, recursion, loops, execution time,
        # memory and algorithmic complexity.
        #
        # A dedicated encoder lets the topic task learn those signals
        # without competing with the difficulty task.
        # ============================================================
        topic_branch = layers.Dense(
            128,
            activation="relu",
            name="topic_dense_1"
        )(inputs)

        topic_branch = layers.BatchNormalization(
            name="topic_bn_1"
        )(topic_branch)

        topic_branch = layers.Dropout(
            0.10,
            name="topic_dropout_1"
        )(topic_branch)

        topic_branch = layers.Dense(
            64,
            activation="relu",
            name="topic_dense_2"
        )(topic_branch)

        topic_branch = layers.BatchNormalization(
            name="topic_bn_2"
        )(topic_branch)

        topic_branch = layers.Dense(
            32,
            activation="relu",
            name="topic_dense_3"
        )(topic_branch)

        topic_out = layers.Dense(
            self.num_topics,
            activation="softmax",
            name="topic"
        )(topic_branch)

        # ============================================================
        # TASK 4: SUCCESS PREDICTION
        # ============================================================
        success_branch = layers.Dense(
            32,
            activation="relu",
            name="success_dense_1"
        )(shared)

        success_branch = layers.Dropout(
            0.15,
            name="success_dropout"
        )(success_branch)

        success_out = layers.Dense(
            1,
            activation="sigmoid",
            name="success"
        )(success_branch)

        # ============================================================
        # MULTI-OUTPUT MODEL
        # ============================================================
        self.model = Model(
            inputs=inputs,
            outputs=[
                difficulty_out,
                skill_out,
                topic_out,
                success_out
            ],
            name="CompetitiveProgrammingModel"
        )

        # ============================================================
        # COMPILE
        # ============================================================
        # Topic gets a higher loss weight because it is the weakest
        # task and is now given its own feature-learning branch.
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.0005
            ),
            loss={
                "difficulty": keras.losses.CategoricalCrossentropy(),
                "skill": keras.losses.MeanSquaredError(),
                "topic": keras.losses.CategoricalCrossentropy(),
                "success": keras.losses.BinaryCrossentropy(),
            },
            loss_weights={
                "difficulty": 1.0,
                "skill": 0.7,
                "topic": 2.5,
                "success": 0.8,
            },
            metrics={
                "difficulty": ["accuracy"],
                "skill": [keras.metrics.MeanAbsoluteError(name="mae")],
                "topic": ["accuracy"],
                "success": ["accuracy"],
            },
        )

        return self.model

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler on training data.
        """
        return self.scaler.fit_transform(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: dict,
        X_val: np.ndarray,
        y_val: dict,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
    ):
        """
        Train the multi-task model.
        """

        # Fit scaler ONLY on training data.
        X_train_scaled = self.prepare_data(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    restore_best_weights=True,
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1,
                ),
            ],
        )

        return history

    def predict(self, X: np.ndarray) -> dict:
        """
        Make predictions on new data.

        Returns:
            dict with difficulty, skill, topic and success.
        """

        X_scaled = self.scaler.transform(X)

        (
            difficulty_pred,
            skill_pred,
            topic_pred,
            success_pred,
        ) = self.model.predict(X_scaled, verbose=0)

        return {
            "difficulty": difficulty_pred,
            "skill": (skill_pred * 100).flatten(),
            "topic": topic_pred,
            "success": success_pred.flatten(),
        }

    def save(self, model_dir: str = "models"):
        """
        Save model and scaler.
        """

        os.makedirs(model_dir, exist_ok=True)

        self.model.save(
            f"{model_dir}/cp_model.h5"
        )

        joblib.dump(
            self.scaler,
            f"{model_dir}/scaler.pkl"
        )

        print(
            f"✅ Model saved to {model_dir}/"
        )

    def load(self, model_dir: str = "models"):
        """
        Load pre-trained model and scaler.
        """

        self.model = keras.models.load_model(
            f"{model_dir}/cp_model.h5"
        )

        self.scaler = joblib.load(
            f"{model_dir}/scaler.pkl"
        )

        print(
            f"✅ Model loaded from {model_dir}/"
        )

    def get_summary(self):
        """
        Print model architecture.
        """

        if self.model is not None:
            self.model.summary()
        else:
            print(
                "Model not built yet. "
                "Call build_model() first."
            )


# ====================================================================
# TRAINING FUNCTION
# ====================================================================

def train_model_from_data(
    solutions_df,
    problems_data
):
    """
    Train model using generated data.

    Validation is split by problem_id so that all five solutions
    belonging to one programming problem remain in the same split.
    """

    from sklearn.model_selection import GroupShuffleSplit

    print(
        "\n" + "=" * 60
    )
    print(
        "PREPARING DATA FOR TRAINING"
    )
    print(
        "=" * 60
    )

    # ================================================================
    # FEATURES
    # ================================================================
    feature_cols = [
        "lines_of_code",
        "cyclomatic_complexity",
        "num_functions",
        "has_recursion",
        "num_loops",
        "execution_time",
        "memory_used_mb",
        "time_to_solve_minutes",
        "algorithmic_complexity",
    ]

    X = solutions_df[
        feature_cols
    ].values.astype(np.float32)

    # ================================================================
    # DIFFICULTY LABEL
    # ================================================================
    difficulty_map = {
        "Easy": [1, 0, 0],
        "Medium": [0, 1, 0],
        "Hard": [0, 0, 1],
    }

    y_difficulty = np.array(
        [
            difficulty_map[d]
            for d in solutions_df["difficulty"]
        ],
        dtype=np.float32,
    )

    # ================================================================
    # SKILL LABEL
    # ================================================================
    # This remains exactly the same proxy used by the project:
    # execution time -> skill estimate.
    max_execution_time = (
        solutions_df["execution_time"].max()
    )

    y_skill = (
        1
        - (
            solutions_df["execution_time"]
            / max_execution_time
        )
    ) * 100

    y_skill = (
        y_skill.values
        .reshape(-1, 1)
        / 100
    ).astype(np.float32)

    # ================================================================
    # TOPIC LABEL
    # ================================================================
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
        "Stack",
    ]

    topic_map = {
        topic: np.eye(
            len(topics),
            dtype=np.float32
        )[i]
        for i, topic in enumerate(topics)
    }

    y_topic = np.array(
        [
            topic_map[t]
            for t in solutions_df["topic"]
        ],
        dtype=np.float32,
    )

    # ================================================================
    # SUCCESS LABEL
    # ================================================================
    y_success = (
        solutions_df["passes_all_tests"]
        .values
        .reshape(-1, 1)
        .astype(np.float32)
    )

    # ================================================================
    # GROUP-BASED TRAIN / VALIDATION SPLIT
    # ================================================================
    # This is the important leakage fix:
    # all 5 solutions from the same problem stay together.
    # ================================================================
    groups = (
        solutions_df["problem_id"]
        .values
    )

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        gss.split(
            X,
            y_difficulty,
            groups=groups,
        )
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

    print(
        f"Training set size: {len(X_train)}"
    )
    print(
        f"Validation set size: {len(X_val)}"
    )
    print(
        f"Training problems: "
        f"{len(np.unique(groups[train_idx]))}"
    )
    print(
        f"Validation problems: "
        f"{len(np.unique(groups[val_idx]))}"
    )

    # ================================================================
    # BUILD MODEL
    # ================================================================
    print(
        "\n" + "=" * 60
    )
    print(
        "BUILDING MODEL"
    )
    print(
        "=" * 60
    )

    model_wrapper = CompetitiveProgrammingModel(
        input_features=len(feature_cols),
        num_topics=len(topics),
    )

    model_wrapper.build_model()
    model_wrapper.get_summary()

    # ================================================================
    # TRAIN
    # ================================================================
    print(
        "\n" + "=" * 60
    )
    print(
        "TRAINING MODEL"
    )
    print(
        "=" * 60
    )

    train_targets = {
        "difficulty": y_diff_train,
        "skill": y_skill_train,
        "topic": y_topic_train,
        "success": y_success_train,
    }

    val_targets = {
        "difficulty": y_diff_val,
        "skill": y_skill_val,
        "topic": y_topic_val,
        "success": y_success_val,
    }

    history = model_wrapper.train(
        X_train,
        train_targets,
        X_val,
        val_targets,
        epochs=100,
        batch_size=32,
        verbose=1,
    )

    print(
        "\n✅ Training complete!"
    )

    # ================================================================
    # SAVE
    # ================================================================
    model_wrapper.save()

    return (
        model_wrapper,
        history,
    )


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":

    from data_generator import CPDatasetGenerator

    print(
        "🔄 Generating dataset..."
    )

    generator = CPDatasetGenerator(
        num_problems=500
    )

    (
        solutions_df,
        problems,
        students,
    ) = generator.save_dataset()

    # Train model
    model, history = train_model_from_data(
        solutions_df,
        problems
    )

    # ================================================================
    # TEST PREDICTION
    # ================================================================
    print(
        "\n" + "=" * 60
    )
    print(
        "TEST PREDICTION"
    )
    print(
        "=" * 60
    )

    test_features = [
        "lines_of_code",
        "cyclomatic_complexity",
        "num_functions",
        "has_recursion",
        "num_loops",
        "execution_time",
        "memory_used_mb",
        "time_to_solve_minutes",
        "algorithmic_complexity",
    ]

    test_sample = (
        solutions_df[
            test_features
        ]
        .iloc[:3]
        .values
    )

    predictions = model.predict(
        test_sample
    )

    topic_names = [
        "Array",
        "String",
        "DP",
        "Graph",
        "Tree",
        "Greedy",
        "Math",
        "Heap",
        "HashTable",
        "Stack",
    ]

    for i, pred in enumerate(
        predictions["difficulty"]
    ):

        difficulty_class = [
            "Easy",
            "Medium",
            "Hard",
        ][np.argmax(pred)]

        skill = predictions["skill"][i]

        success_prob = (
            predictions["success"][i]
        )

        topic_index = np.argmax(
            predictions["topic"][i]
        )

        topic_name = topic_names[
            topic_index
        ]

        topic_confidence = (
            np.max(
                predictions["topic"][i]
            ) * 100
        )

        print(
            f"\nSample {i + 1}:"
        )

        print(
            f"  Predicted Difficulty: "
            f"{difficulty_class} "
            f"({np.max(pred) * 100:.1f}%)"
        )

        print(
            f"  Predicted Topic: "
            f"{topic_name} "
            f"({topic_confidence:.1f}%)"
        )

        print(
            f"  Estimated Skill: "
            f"{skill:.1f}/100"
        )

        print(
            f"  Success Probability: "
            f"{success_prob * 100:.1f}%"
        )
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os


class CPDatasetGenerator:

    def __init__(
        self,
        num_problems: int = 500,
        seed: int = 42
    ):
        self.num_problems = num_problems

        np.random.seed(seed)

        self.topics = [
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

        self.difficulties = [
            "Easy",
            "Medium",
            "Hard"
        ]

    # ============================================================
    # CODE FEATURE GENERATION
    # ============================================================

    def generate_code_features(
        self,
        difficulty: str,
        topic: str
    ) -> Dict:

        difficulty_multiplier = {
            "Easy": 1.0,
            "Medium": 1.5,
            "Hard": 2.2
        }

        base_loc = {
            "Easy": 50,
            "Medium": 100,
            "Hard": 180
        }

        topic_profiles = {

            "Array": {
                "complexity": 0.9,
                "functions": 0,
                "recursion": 0.01,
                "loops": 2.0
            },

            "String": {
                "complexity": 1.1,
                "functions": 1,
                "recursion": 0.02,
                "loops": 1.0
            },

            "DP": {
                "complexity": 2.4,
                "functions": 2,
                "recursion": 0.45,
                "loops": 2.2
            },

            "Graph": {
                "complexity": 2.7,
                "functions": 3,
                "recursion": 0.55,
                "loops": 2.6
            },

            "Tree": {
                "complexity": 2.2,
                "functions": 2,
                "recursion": 0.75,
                "loops": 1.5
            },

            "Greedy": {
                "complexity": 1.0,
                "functions": 0,
                "recursion": 0.01,
                "loops": 0.6
            },

            "Math": {
                "complexity": 1.5,
                "functions": 1,
                "recursion": 0.08,
                "loops": 0.4
            },

            "Heap": {
                "complexity": 1.8,
                "functions": 2,
                "recursion": 0.03,
                "loops": 1.4
            },

            "HashTable": {
                "complexity": 1.3,
                "functions": 2,
                "recursion": 0.01,
                "loops": 0.8
            },

            "Stack": {
                "complexity": 1.4,
                "functions": 1,
                "recursion": 0.20,
                "loops": 1.0
            }
        }

        profile = topic_profiles[topic]
        multiplier = difficulty_multiplier[difficulty]

        # --------------------------------------------------------
        # Lines of code
        # --------------------------------------------------------

        loc = int(
            base_loc[difficulty]
            * np.random.normal(1.0, 0.15)
            * (0.9 + 0.1 * profile["complexity"])
        )

        loc = max(20, loc)

        # --------------------------------------------------------
        # Cyclomatic complexity
        # --------------------------------------------------------

        cyclomatic = int(
            4 * multiplier
            + profile["complexity"]
            + np.random.normal(0, 1.5)
        )

        cyclomatic = max(1, cyclomatic)

        # --------------------------------------------------------
        # Number of functions
        # --------------------------------------------------------

        num_functions = int(
            1
            + profile["functions"]
            + np.random.poisson(
                max(0.5, multiplier - 0.5)
            )
        )

        num_functions = max(1, num_functions)

        # --------------------------------------------------------
        # Recursion
        # --------------------------------------------------------

        difficulty_recursion = {
            "Easy": 0.15,
            "Medium": 0.35,
            "Hard": 0.60
        }

        recursion_probability = (
            difficulty_recursion[difficulty] * 0.5
            + profile["recursion"] * 0.5
        )

        has_recursion = int(
            np.random.rand() < recursion_probability
        )

        # --------------------------------------------------------
        # Number of loops
        # --------------------------------------------------------

        num_loops = int(
            np.random.gamma(2, 0.7)
            * multiplier
            * profile["loops"]
        )

        num_loops = max(0, num_loops)

        # --------------------------------------------------------
        # Algorithmic complexity
        # --------------------------------------------------------

        algorithmic_complexity = (
            cyclomatic
            + 0.5 * num_functions
            + 0.8 * num_loops
            + 1.5 * has_recursion
            + np.random.normal(0, 1)
        )

        algorithmic_complexity = max(
            0.5,
            algorithmic_complexity
        )

        return {
            "lines_of_code": loc,
            "cyclomatic_complexity": cyclomatic,
            "num_functions": num_functions,
            "has_recursion": has_recursion,
            "num_loops": num_loops,
            "algorithmic_complexity": round(
                algorithmic_complexity,
                2
            )
        }

    # ============================================================
    # STUDENT SOLUTION GENERATION
    # ============================================================

    def generate_student_solution(
        self,
        problem_id: int,
        difficulty: str,
        topic: str
    ) -> Dict:

        features = self.generate_code_features(
            difficulty,
            topic
        )

        # --------------------------------------------------------
        # Execution time
        # --------------------------------------------------------

        topic_time_factor = {
            "Array": 0.9,
            "String": 1.0,
            "DP": 1.5,
            "Graph": 1.8,
            "Tree": 1.4,
            "Greedy": 0.8,
            "Math": 1.1,
            "Heap": 1.3,
            "HashTable": 1.6,
            "Stack": 0.7
        }

        difficulty_time = {
            "Easy": 1.0,
            "Medium": 3.0,
            "Hard": 8.0
        }

        exec_time = (
            difficulty_time[difficulty]
            * topic_time_factor[topic]
            + np.random.normal(0, 0.4)
        )

        exec_time = max(
            0.1,
            exec_time
        )

        # --------------------------------------------------------
        # Memory usage
        # --------------------------------------------------------

        topic_memory_factor = {
            "Array": 0.9,
            "String": 1.0,
            "DP": 1.8,
            "Graph": 2.2,
            "Tree": 1.5,
            "Greedy": 0.8,
            "Math": 0.7,
            "Heap": 1.6,
            "HashTable": 1.9,
            "Stack": 1.1
        }

        memory = (
            10
            + int(
                25
                * topic_memory_factor[topic]
                * np.random.uniform(0.7, 1.3)
            )
        )

        # --------------------------------------------------------
        # Probability of passing
        # --------------------------------------------------------

        pass_rate = {
            "Easy": 0.90,
            "Medium": 0.65,
            "Hard": 0.40
        }

        probability = pass_rate[difficulty]

        if topic in [
            "Graph",
            "DP",
            "Tree"
        ]:
            probability -= 0.05

        elif topic in [
            "Greedy",
            "Array",
            "Stack"
        ]:
            probability += 0.03

        probability = np.clip(
            probability,
            0.15,
            0.95
        )

        passes_tests = int(
            np.random.rand() < probability
        )

        # --------------------------------------------------------
        # Time required to solve
        # --------------------------------------------------------

        solve_time_base = {
            "Easy": 5,
            "Medium": 20,
            "Hard": 45
        }

        solve_time = (
            solve_time_base[difficulty]
            * topic_time_factor[topic]
            + np.random.normal(0, 4)
        )

        solve_time = max(
            1,
            solve_time
        )

        return {
            "problem_id": problem_id,
            "topic": topic,
            "difficulty": difficulty,
            **features,
            "execution_time": round(
                exec_time,
                2
            ),
            "memory_used_mb": memory,
            "passes_all_tests": passes_tests,
            "time_to_solve_minutes": round(
                solve_time,
                1
            )
        }

    # ============================================================
    # DATASET GENERATION
    # ============================================================

    def generate_dataset(
        self
    ) -> Tuple[pd.DataFrame, List[Dict]]:

        solutions = []
        problems = []

        for problem_id in range(
            self.num_problems
        ):

            # Rotate through topics so every topic
            # receives approximately the same number
            # of problems.

            topic = self.topics[
                problem_id % len(self.topics)
            ]

            difficulty = np.random.choice(
                self.difficulties
            )

            problem = {
                "id": problem_id,

                "title": (
                    f"{topic} Problem "
                    f"{problem_id}"
                ),

                "topic": topic,

                "difficulty": difficulty,

                "difficulty_score": {
                    "Easy": 1,
                    "Medium": 2,
                    "Hard": 3
                }[difficulty],

                "description": (
                    f"Solve this "
                    f"{difficulty} "
                    f"{topic} problem"
                )
            }

            problems.append(problem)

            # Generate 5 solutions for every problem.
            # This is why we MUST split by problem_id
            # during model evaluation.

            for _ in range(5):

                solution = (
                    self.generate_student_solution(
                        problem_id,
                        difficulty,
                        topic
                    )
                )

                solutions.append(solution)

        solutions_df = pd.DataFrame(
            solutions
        )

        return (
            solutions_df,
            problems
        )

    # ============================================================
    # STUDENT SUBMISSION HISTORY
    # ============================================================

    def create_student_submission_history(
        self,
        num_students: int = 20
    ) -> Dict:

        students = {}

        for student_id in range(
            num_students
        ):

            num_submissions = np.random.randint(
                10,
                30
            )

            student_skill = np.random.randint(
                30,
                95
            )

            submissions = []
            solved_problems = []

            for _ in range(
                num_submissions
            ):

                difficulty_bias = (
                    student_skill / 100
                )

                problem_difficulty = min(
                    3,
                    max(
                        1,
                        int(
                            np.random.normal(
                                difficulty_bias * 3,
                                1
                            )
                        )
                    )
                )

                difficulty_map = {
                    1: "Easy",
                    2: "Medium",
                    3: "Hard"
                }

                difficulty = difficulty_map[
                    problem_difficulty
                ]

                topic = np.random.choice(
                    self.topics
                )

                problem_id = np.random.randint(
                    0,
                    self.num_problems
                )

                solution = (
                    self.generate_student_solution(
                        problem_id,
                        difficulty,
                        topic
                    )
                )

                solution["student_id"] = (
                    student_id
                )

                solution[
                    "student_skill_estimate"
                ] = student_skill

                submissions.append(
                    solution
                )

                if solution[
                    "passes_all_tests"
                ]:
                    solved_problems.append(
                        problem_id
                    )

            students[
                f"student_{student_id}"
            ] = {
                "skill_level": student_skill,

                "submissions": submissions,

                "solved_problems": list(
                    set(solved_problems)
                ),

                "num_submissions": (
                    num_submissions
                )
            }

        return students

    # ============================================================
    # SAVE DATASET
    # ============================================================

    def save_dataset(
        self,
        output_dir: str = "data"
    ):

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        solutions_df, problems = (
            self.generate_dataset()
        )

        students = (
            self.create_student_submission_history()
        )

        # --------------------------------------------------------
        # Save problems
        # --------------------------------------------------------

        with open(
            f"{output_dir}/problems.json",
            "w"
        ) as f:

            json.dump(
                problems,
                f,
                indent=2
            )

        # --------------------------------------------------------
        # Save solutions
        # --------------------------------------------------------

        solutions_df.to_csv(
            f"{output_dir}/solutions.csv",
            index=False
        )

        # --------------------------------------------------------
        # Save student history
        # --------------------------------------------------------

        with open(
            f"{output_dir}/student_history.json",
            "w"
        ) as f:

            json.dump(
                students,
                f,
                indent=2
            )

        print(
            f"✅ Dataset saved to "
            f"{output_dir}/"
        )

        return (
            solutions_df,
            problems,
            students
        )


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print(
        "🔄 Generating dataset..."
    )

    generator = CPDatasetGenerator(
        num_problems=500
    )

    solutions_df, problems, students = (
        generator.save_dataset()
    )

    print(
        "✅ Done!"
    )
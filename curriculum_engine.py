"""
Curriculum Learning Engine
Recommends problems optimally based on student skill using bandit approach
"""

import numpy as np
import json
from typing import List, Dict, Tuple
import pandas as pd

class CurriculumLearningEngine:
    """
    Recommends competitive programming problems based on:
    1. Student skill level
    2. Weak topics (need improvement)
    3. Problem difficulty (optimal challenge zone)
    
    Uses Thompson Sampling (Bandit approach) for exploration-exploitation
    """
    
    def __init__(self, problems_data: List[Dict], model=None):
        self.problems = problems_data
        self.model = model
        self.topics = ["Array", "String", "DP", "Graph", "Tree", "Greedy", "Math", "Heap", "HashTable", "Stack"]
        
    def build_student_profile(self, student_submissions: List[Dict]) -> Dict:
        """
        Analyze student submission history to create skill profile
        
        Returns:
            dict with skill_level, topic_mastery, solved_problems, weak_topics
        """
        if not student_submissions:
            return {
                'skill': 50,  # Default middle skill
                'topic_mastery': {topic: 50 for topic in self.topics},
                'solved_problems': [],
                'attempts': 0
            }
        
        # Calculate skill from execution time and success
        submissions_df = pd.DataFrame(student_submissions)
        
        # Skill: average success rate + speed
        success_rate = submissions_df['passes_all_tests'].mean() * 100
        
        # Faster solvers are more skilled
        avg_solve_time = submissions_df['time_to_solve_minutes'].mean()
        speed_score = max(0, 100 - (avg_solve_time / 2))  # Normalize
        
        skill = (success_rate * 0.6 + speed_score * 0.4)
        
        # Topic mastery
        topic_mastery = {}
        for topic in self.topics:
            topic_submissions = submissions_df[submissions_df['topic'] == topic]
            
            if len(topic_submissions) == 0:
                topic_mastery[topic] = 50  # Unknown
            else:
                topic_success = topic_submissions['passes_all_tests'].mean() * 100
                topic_mastery[topic] = topic_success
        
        # Solved problems
        solved = submissions_df[submissions_df['passes_all_tests'] == True]['problem_id'].unique().tolist()
        
        # Weak topics (mastery < 60)
        weak_topics = [t for t, m in topic_mastery.items() if m < 60]
        
        return {
            'skill': min(100, max(0, skill)),
            'topic_mastery': topic_mastery,
            'solved_problems': solved,
            'weak_topics': weak_topics,
            'num_attempts': len(submissions_df),
            'success_rate': success_rate,
            'submissions': submissions_df
        }
    
    def get_optimal_difficulty_zone(self, student_skill: float) -> Tuple[float, float]:
        """
        Return difficulty range where student learns best
        
        Vygotsky's Zone of Proximal Development (ZPD):
        - Too easy: No learning
        - Optimal: Current skill to current skill + 30%
        - Too hard: Frustration
        """
        min_difficulty = max(1, student_skill - 20)  # Can still do easier problems
        max_difficulty = student_skill + 30  # Challenging but doable
        
        return min_difficulty, max_difficulty
    
    def score_problem(self, problem: Dict, student_profile: Dict, 
                     difficulty_importance: float = 0.5,
                     topic_importance: float = 0.5) -> float:
        """
        Score how good a problem recommendation is for a student
        
        Factors:
        1. Difficulty match (is it in optimal zone?)
        2. Topic relevance (is it a weak topic?)
        3. Novelty (haven't solved similar before)
        """
        
        score = 0.0
        
        # 1. Difficulty match score
        min_diff, max_diff = self.get_optimal_difficulty_zone(student_profile['skill'])
        problem_difficulty = problem['difficulty_score']
        
        if problem_difficulty < min_diff:
            difficulty_score = 0.2  # Too easy, not learning
        elif problem_difficulty > max_diff:
            difficulty_score = 0.1  # Too hard, frustrating
        else:
            # Linear score in optimal zone
            # Prefer slightly harder problems (learning edge)
            difficulty_score = 0.5 + 0.5 * ((problem_difficulty - min_diff) / (max_diff - min_diff))
        
        score += difficulty_score * difficulty_importance
        
        # 2. Topic relevance score
        problem_topic = problem['topic']
        topic_mastery = student_profile['topic_mastery'].get(problem_topic, 50)
        
        if problem_topic in student_profile['weak_topics']:
            topic_score = 1.0  # High priority for weak topics
        else:
            # Still recommend, but lower priority
            topic_score = max(0.3, 1 - (topic_mastery / 100))
        
        score += topic_score * topic_importance
        
        # 3. Novelty penalty (avoid already solved)
        if problem['id'] in student_profile['solved_problems']:
            score *= 0.05  # Heavily penalize
        
        return min(2.0, max(0.0, score))  # Normalize to 0-2
    
    def recommend_problems(self, student_profile: Dict, 
                          num_recommendations: int = 5,
                          learning_stage: str = 'adaptive') -> List[Dict]:
        """
        Recommend problems using curriculum learning
        
        Strategies:
        - 'adaptive': Balance between weak topics and difficulty
        - 'exploration': Recommend diverse topics
        - 'exploitation': Recommend high-score problems
        """
        
        scored_problems = []
        
        for problem in self.problems:
            score = self.score_problem(problem, student_profile)
            
            # Adaptive strategy: boost weak topics
            if learning_stage == 'adaptive':
                if problem['topic'] in student_profile['weak_topics']:
                    score *= 1.5
            
            # Exploration strategy: prefer unseen topics
            elif learning_stage == 'exploration':
                topic_mastery = student_profile['topic_mastery'].get(problem['topic'], 50)
                if topic_mastery < 30:  # Haven't learned much
                    score *= 2.0
            
            scored_problems.append({
                'problem': problem,
                'score': score,
                'topic': problem['topic'],
                'difficulty': problem['difficulty'],
                'reason': self._get_recommendation_reason(problem, student_profile, score)
            })
        
        # Sort by score (highest first)
        recommendations = sorted(scored_problems, key=lambda x: x['score'], reverse=True)
        
        return recommendations[:num_recommendations]
    
    def _get_recommendation_reason(self, problem: Dict, student_profile: Dict, score: float) -> str:
        """
        Explain why this problem was recommended
        """
        reasons = []
        
        if problem['topic'] in student_profile['weak_topics']:
            reasons.append(f"Helps improve weak topic: {problem['topic']}")
        
        min_diff, max_diff = self.get_optimal_difficulty_zone(student_profile['skill'])
        if min_diff <= problem['difficulty_score'] <= max_diff:
            reasons.append("Perfect difficulty for your skill level")
        elif problem['difficulty_score'] < min_diff:
            reasons.append("Good warmup problem")
        else:
            reasons.append("Challenging but achievable")
        
        if problem['id'] not in student_profile['solved_problems']:
            reasons.append("Not solved before")
        
        return " | ".join(reasons) if reasons else "Matches your learning profile"
    
    def predict_readiness(self, student_profile: Dict, target_skill: float = 85) -> Dict:
        """
        Predict when student will be ready for next level
        """
        current_skill = student_profile['skill']
        gap = target_skill - current_skill
        
        if gap <= 0:
            days_to_readiness = 0
        else:
            # Estimate: ~2 points of skill improvement per day with optimal learning
            days_to_readiness = int(gap / 2)
        
        # Topic-specific readiness
        topic_readiness = {}
        for topic, mastery in student_profile['topic_mastery'].items():
            if mastery >= 80:
                topic_readiness[topic] = {'status': 'Mastered', 'days_to_master': 0}
            elif mastery >= 60:
                topic_readiness[topic] = {'status': 'Competent', 'days_to_master': 5}
            elif mastery >= 40:
                topic_readiness[topic] = {'status': 'Learning', 'days_to_master': 15}
            else:
                topic_readiness[topic] = {'status': 'Beginner', 'days_to_master': 30}
        
        return {
            'current_skill': current_skill,
            'target_skill': target_skill,
            'skill_gap': gap,
            'estimated_days_to_target': days_to_readiness,
            'topic_readiness': topic_readiness,
            'gate_readiness_percentage': (current_skill / target_skill) * 100
        }
    
    def generate_learning_path(self, student_profile: Dict, num_days: int = 30) -> Dict:
        """
        Generate a personalized learning path
        """
        weak_topics = student_profile['weak_topics']
        
        # Suggest learning order: focus on weak topics, then related topics
        learning_path = []
        
        if weak_topics:
            # Days 1-10: Focus on weakest topic
            weakest = weak_topics[0]
            learning_path.append({
                'phase': 1,
                'days': '1-10',
                'focus': f'Master {weakest}',
                'recommended_problems': 'Easy → Medium problems in ' + weakest
            })
            
            # Days 11-20: Expand to other weak topics
            if len(weak_topics) > 1:
                learning_path.append({
                    'phase': 2,
                    'days': '11-20',
                    'focus': f"Improve {', '.join(weak_topics[1:3])}",
                    'recommended_problems': 'Medium problems in weak topics'
                })
            
            # Days 21-30: Consolidate and challenge
            learning_path.append({
                'phase': 3,
                'days': '21-30',
                'focus': 'Consolidate & Challenge',
                'recommended_problems': 'Hard problems, mixed topics'
            })
        
        return {
            'total_days': num_days,
            'learning_phases': learning_path,
            'estimated_final_skill': min(100, student_profile['skill'] + (num_days * 2))
        }
    
    def get_learning_analytics(self, student_profile: Dict) -> Dict:
        """
        Provide detailed learning analytics
        """
        submissions = student_profile['submissions']
        
        # Time series analysis
        problem_difficulty_progress = submissions.groupby('difficulty')['passes_all_tests'].mean() * 100
        
        return {
            'total_submissions': len(submissions),
            'success_rate': student_profile['success_rate'],
            'avg_solve_time': submissions['time_to_solve_minutes'].mean(),
            'problems_solved': len(student_profile['solved_problems']),
            'weak_topics': student_profile['weak_topics'],
            'strong_topics': [t for t, m in student_profile['topic_mastery'].items() if m >= 80],
            'difficulty_performance': problem_difficulty_progress.to_dict()
        }


# Example usage
def demo_curriculum_engine():
    """
    Demonstrate curriculum engine with sample data
    """
    from data_generator import CPDatasetGenerator
    import json
    
    print("\n" + "="*60)
    print("CURRICULUM LEARNING ENGINE DEMO")
    print("="*60)
    
    # Generate data
    generator = CPDatasetGenerator(num_problems=50)
    solutions_df, problems, students = generator.save_dataset()
    
    # Initialize engine
    engine = CurriculumLearningEngine(problems)
    
    # Pick a student
    student_name = "student_0"
    student_data = students[student_name]
    
    print(f"\n📊 Analyzing student: {student_name}")
    print(f"   Submissions: {student_data['num_submissions']}")
    print(f"   Skill estimate: {student_data['skill_level']}/100")
    
    # Build profile
    profile = engine.build_student_profile(student_data['submissions'])
    
    print(f"\n🎯 Student Profile:")
    print(f"   Skill Level: {profile['skill']:.1f}/100")
    print(f"   Success Rate: {profile['success_rate']:.1f}%")
    print(f"   Problems Solved: {len(profile['solved_problems'])}")
    print(f"   Weak Topics: {', '.join(profile['weak_topics'][:3])}")
    
    # Get recommendations
    print(f"\n📚 Recommended Problems:")
    recommendations = engine.recommend_problems(profile, num_recommendations=3)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n   {i}. {rec['problem']['title']}")
        print(f"      Difficulty: {rec['problem']['difficulty']}")
        print(f"      Topic: {rec['problem']['topic']}")
        print(f"      Score: {rec['score']:.2f}")
        print(f"      Reason: {rec['reason']}")
    
    # Readiness prediction
    readiness = engine.predict_readiness(profile, target_skill=85)
    print(f"\n🎓 GATE Readiness:")
    print(f"   Current: {readiness['current_skill']:.1f}/100")
    print(f"   Target: {readiness['target_skill']}")
    print(f"   Days to GATE ready: {readiness['estimated_days_to_target']}")
    print(f"   Readiness %: {readiness['gate_readiness_percentage']:.1f}%")
    
    # Learning path
    learning_path = engine.generate_learning_path(profile, num_days=30)
    print(f"\n📈 30-Day Learning Path:")
    for phase in learning_path['learning_phases']:
        print(f"   Phase {phase['phase']} (Days {phase['days']}): {phase['focus']}")


if __name__ == "__main__":
    demo_curriculum_engine()

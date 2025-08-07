import dataclasses
import json
import os
import numpy as np

def a(w):
    return np.array(w)

@dataclasses.dataclass
class Data_ACS:
    num_materials = 120
    num_concepts = 20
    num_students = 30
    script_dir = os.path.dirname(os.path.abspath(__file__))

    course_data_path = os.path.join(script_dir, 'course_data.txt')

    with open(course_data_path, 'r', encoding='utf-8') as file:
        course_data = json.load(file)

    m_coverage = course_data["materials_coverage"]
    m_coverage = a(m_coverage)
    m_coverage_shape = m_coverage.shape
    m_difficulty = course_data["materials_difficulty"]
    m_difficulty = a(m_difficulty)
    m_difficulty_shape = m_difficulty.shape
    m_time = course_data["materials_time"]
    m_time = a(m_time)
    m_time_shape = m_time.shape
    m_preferences = course_data["materials_preferences"]
    m_preferences = a(m_preferences)
    m_preferences_shape = m_preferences.shape

    script_dir = os.path.dirname(os.path.abspath(__file__))

    course_data_path = os.path.join(script_dir, 'students_data.txt')

    with open(course_data_path, 'r', encoding='utf-8') as file:
        students_data = json.load(file)

    s_goals = students_data["student_goals"]
    s_goals = a(s_goals)
    s_goals_shape = s_goals.shape
    s_ability = students_data["student_ability"]
    s_ability = a(s_ability)
    s_ability_shape = s_ability.shape
    s_time_limits = students_data["student_time_limits"]
    s_time_limits = a(s_time_limits)
    s_time_limits_shape = s_time_limits.shape
    s_preferences = students_data["student_preferences"]
    s_preferences = a(s_preferences)
    s_preferences_shape = s_preferences.shape

@dataclasses.dataclass
class Data_Algorithm:
    SearchAgents = 20
    dim = Data_ACS.num_materials * Data_ACS.num_students
    MaxIter = 50
    ub = np.ones((1, dim))
    lb = np.zeros((1, dim))
    capacity = 10






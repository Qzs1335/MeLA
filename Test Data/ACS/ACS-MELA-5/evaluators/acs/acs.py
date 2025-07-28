from typing import Any
import numpy as np
from data.data import Data_ACS,Data_Algorithm


class AcsEvaluation():
    def __init__(self, **kwargs):
        self.data = Data_ACS()
        self.data_al = Data_Algorithm()


    def covers_goals(self, material_idx, student_goals, material_coverage):
        return np.all(student_goals[material_coverage[material_idx, :] == 1] == 1)

    def f_obj(self, X):
        X = np.reshape(X, (self.data.num_students, -1))
        X = np.where(X < 0.5, 0, 1)

        fitness = 0
        re = []
        vio = []
        vio_2 = []

        recommended_materials = [np.where(X[i, :] == 1)[0] for i in range(X.shape[0])]
        sorted_recommended_materials = [None] * len(recommended_materials)

        max_high_priority = 3
        max_medium_priority = 6
        max_challenge_materials = 1

        for i, materials in enumerate(recommended_materials):
            if len(materials) > 0:
                difficulties = [self.data.m_difficulty[idx] for idx in materials]
                sorted_materials = [x for _, x in sorted(zip(difficulties, materials))]

                high_priority = []
                medium_priority = []
                challenge = []

                for material_idx in sorted_materials:
                    difficulty = self.data.m_difficulty[material_idx]
                    if (difficulty <= self.data.s_ability[i] and
                            self.covers_goals(material_idx, np.array(self.data.s_goals[i]),
                                              np.array(self.data.m_coverage))):
                        if len(high_priority) < max_high_priority:
                            high_priority.append(material_idx)
                    elif (difficulty <= self.data.s_ability[i] and
                          not self.covers_goals(material_idx, np.array(self.data.s_goals[i]),
                                                np.array(self.data.m_coverage))):
                        if len(medium_priority) < max_medium_priority:
                            medium_priority.append(material_idx)
                    elif difficulty > self.data.s_ability[i]:
                        if len(challenge) < max_challenge_materials:
                            challenge.append(material_idx)

                sorted_recommended_materials[i] = high_priority + medium_priority + challenge
            else:
                sorted_recommended_materials[i] = []

        temp_re = []
        for i in range(self.data.num_students):
            recommended = sorted_recommended_materials[i]
            for material_idx in recommended:
                if self.data.m_difficulty[material_idx] < self.data.s_ability[i]:
                    temp_re.append(material_idx)
            re.append(temp_re.copy())
            temp_re.clear()

        val = np.zeros((self.data.num_students, self.data.num_concepts))
        for i in range(self.data.num_students):
            wt = re[i]
            if len(wt) > 0:
                for material_idx in wt:
                    material_idx = int(material_idx)
                    val[i, :] += self.data.m_coverage[material_idx]

        val_temp = val - self.data.s_goals
        num_not_cover = np.sum(val_temp == -1)
        fitness += num_not_cover * 1e4
        vio.append(num_not_cover)
        n = 0

        for i in range(self.data.num_students):
            for j in range(self.data.num_concepts):
                if val_temp[i, j] >= 1:
                    fitness += val_temp[i, j] * 0.25
                    n += val_temp[i, j]
        vio_2.append(n)

        temp_v = 0
        for i in range(self.data.num_students):
            wt_1 = re[i]
            if len(wt_1) > 0:
                times = [self.data.m_time[material_idx] for material_idx in wt_1]
                total_time = sum(sum(sublist) for sublist in times)
                if (total_time < self.data.s_time_limits[i][0] or
                        total_time > self.data.s_time_limits[i][1]):
                    temp_v += 1

        fitness += temp_v * 1000
        vio.append(temp_v)

        n_2 = 0
        for i in range(self.data.num_students):
            recommended = sorted_recommended_materials[i]
            if len(recommended) > 0:
                for material_idx in recommended:
                    pref_diff = abs(np.mean(self.data.s_preferences[i]) -
                                    np.mean(self.data.m_preferences[material_idx]))
                    fitness += pref_diff * 0.25
                    n_2 += pref_diff

        vio_2.append(n_2)

        return fitness

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        return self.evaluate(priority=callable_func)

    def evaluate(self, priority: callable) -> list[Any]:
        SearchAgents_no = self.data_al.SearchAgents
        Max_iter = self.data_al.MaxIter
        dim = self.data_al.dim
        lb = np.array(self.data_al.lb)
        ub = np.array(self.data_al.ub)

        ub_array = np.array(ub)
        lb_array = np.array(lb)
        if ub_array.size == 1:
            Positions = np.random.rand(SearchAgents_no, dim) * (ub_array - lb_array) + lb_array
        else:
            Positions = np.zeros((SearchAgents_no, dim))
            for i in range(dim):
                ub_i = ub_array[0][1]
                lb_i = lb_array[0][1]
                Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

        S = 2.5
        rg = S
        decay_rate = S / Max_iter

        Best_pos = np.full(dim, np.inf)
        Best_score = np.inf
        Convergence_curve = []

        for t in range(Max_iter):
            for i in range(Positions.shape[0]):
                fitness = self.f_obj(Positions[i, :])
                if fitness < Best_score:
                    Best_score = fitness
                    Best_pos = Positions[i, :].copy()
            Convergence_curve.append(Best_score)

            Positions = priority(Positions, Best_pos, Best_score, rg)

            rg = max(0.1, rg - decay_rate)

        return Convergence_curve
import concurrent.futures
import logging
import os.path
import random
import subprocess
import numpy as np
from omegaconf import DictConfig
from utils.utils import *
from utils.llm_client.base import BaseClient
from evaluators.acs.data.data import Data_ACS, Data_Algorithm
from promoter.promoter import *
from promoter.system_role import *

class ELE:
    def __init__(self,
                 cfg: DictConfig,
                 root_dir: str,
                 llm: BaseClient
                 ):
        self.cfg = cfg
        self.root_dir = root_dir
        self.iteration = 0
        self.function_evals = 0
        self.num_database = self.cfg.problem.problem_name
        self.elitist = None
        self.llm = llm
        self.retry = 0

        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self._data_pb = Data_ACS()
        self._data_al = Data_Algorithm()
        self._num_evolve = 0
        self._results = []
        self._evolve_part = []
        current_dir = os.path.dirname(os.path.abspath(__file__))


        self.problem = self.cfg.problem.problem_name
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type

        logging.info("Problem: " + self.problem)
        # logging.info("Problem code: " + self.problem_code)
        logging.info("Function name: " + self.func_name)

        self.promoter_dir = f"{self.root_dir}/promoter"
        self.output_prompt_file = f"{self.root_dir}/sample_prompt"
        self.output_file = f"{self.root_dir}/evaluators/{self.cfg.problem.problem_name}/gpt.py"
        self.problem_role = problem_role()
        self.system_role = system_role()
        self.error_role = error_role()
        self.e_learning_role = e_learning_role()
        # self.expert_role = expert_role()
        self.print_sample_prompt = True
        self.init_optimize()

    def problem_analyze(self):

        problem_file = f"{self.root_dir}/evaluators/{self.cfg.problem.problem_name}/{self.cfg.problem.problem_name}.py"
        with open(problem_file, 'r',encoding='utf-8',errors='replace') as f:
            problem = f.read()
        user = problem_prompt(problem)
        messages = [{"role": "system", "content": self.problem_role}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt:")

        response = self.llm.multi_chat_completion([messages])[0]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'thoughtbase', 'problem_analyze.txt')
        with open(file_path, 'w', encoding='utf-8', errors='replace') as file:
            file.write(response + '\n')

        return response

    def strip_ansi(self, text: str) -> str:
        """Remove ANSI color escape codes"""
        import re
        return re.sub(r'\x1b\[[0-9;]*[mK]', '', text)

    def extract_complete_error(self, exception: Exception) -> str:
        """Extract the complete error type and message (including subsequent recommendations)"""
        error_str = self.strip_ansi(str(exception))
        lines = [line.strip() for line in error_str.split('\n') if line.strip()]

        if not lines:
            return str(exception)

        # Find the first line containing an error type (such as "NameError:")
        error_type_line = None
        for i, line in enumerate(lines):
            if 'Error:' in line or 'Exception:' in line:
                error_type_line = i
                break

        if error_type_line is None:
            return lines[-1]  # If there is no clear error type, return the last line.

        # Return all content starting from the line with the error type.
        return '\n'.join(lines[error_type_line:])

    def init_optimize(self):
        logging.info("Evaluating init algorithm...")
        logging.info("Analyze the problem...")
        self.problem_a = self.problem_analyze()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'experts', 'scso.txt')
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            self.code_init = file.read()
        file_path = os.path.join(current_dir, 'promoter', 'init_code_empty.txt')
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            code_to_evolve = file.read()

        code_init = extract_code_from_generator(self.code_init)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code_init,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.codebase = self.codebase_evaluate([seed_ind])

        history_path = os.path.join(self.root_dir, 'thoughtbase/thought_history.txt')
        with open(history_path, 'a', encoding='utf-8', errors='replace') as f:
            f.write(self.code_init + '\nThe init score:' + str(self.codebase[0]['obj']) + '\n')

        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.next_gen()
        test_script_stdout = f"{self.root_dir}/codebase/iteration_0/individual_0_obj.txt"
        with open(test_script_stdout, 'r', encoding='utf-8', errors='replace') as f:
            self.init_eval = f.read()
        logging.info(f"Validation script finished. Results are saved in {test_script_stdout}.")

        user = return_promoter_init(self.code_init, self.init_eval, code_to_evolve, self.problem_a)
        messages = [{"role": "system", "content": self.system_role}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt:")

        responses = self.llm.multi_chat_completion([messages], self.cfg.init_pop_size,
                                                             temperature=self.llm.temperature + 0.3)
        codebase = [self.response_to_individual(response, response_id) for response_id, response in
                      enumerate(responses)]
        codebase = self.codebase_evaluate(codebase)

        for individual in codebase:
            with open(history_path, 'a', encoding='utf-8', errors='replace') as f:
                try:
                    f.write('\n' + individual['thought'] + '\nThe score:' + str(individual['obj']))
                except:
                    pass
        self.codebase = codebase
        self.next_gen()

    def response_to_individual(self, response: str, response_id, file_name: str = None) -> dict:
        # **MODIFIED**: response_id can now be a string, e.g., "5_cand0"

        # If a base filename is provided, use it. Otherwise, construct from response_id.
        response_file_name_base = file_name if file_name is not None else f"problem_iter{self.iteration}_response{response_id}"

        with open(response_file_name_base + ".txt", 'w', encoding='utf-8', errors='replace') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)
        if code is not None:
            code = code.replace('v1', 'v2')

        thought = trim_thought_from_response(response)
        if thought is not None:
            thought = thought.replace('{', '').replace('}', '')

        individual = {
            # All filenames are now derived from the unique response_id
            "stdout_filepath": f"problem_iter{self.iteration}_stdout_{response_id}.txt",
            "code_path": f"individual_{response_id}_code.py",
            "code": code,
            "response_id": response_id,  # Store the unique ID
            "thought": thought
        }
        return individual

    def _generate_candidates(self, failures: dict, max_candidates: int) -> list:
        """Generate all candidate solutions in one batch with unique IDs."""
        if not failures:
            return []

        # Prepare all prompts at once
        batch_messages = []
        # **MODIFIED**: Store unique ID and original index for each message
        # The list will contain tuples of: (unique_id, original_index, messages)
        for idx, failure in failures.items():
            for i in range(max_candidates):
                # 1. CREATE A GLOBALLY UNIQUE ID for each candidate
                # This ID includes the original failure index and the candidate number (0, 1, ...)
                unique_candidate_id = f"{idx}_cand{i}"

                user_msg = error_prompt(failure["error_msg"], failure["individual"]["code"])
                messages = [
                    {"role": "system", "content": self.error_role},
                    {"role": "user", "content": user_msg}
                ]
                batch_messages.append((unique_candidate_id, idx, messages))

        # Batch process all LLM requests
        # We only need the messages for the LLM call
        responses = self.llm.multi_chat_completion([msg for _, _, msg in batch_messages], temperature=self.llm.temperature + 0.3)

        # Build all candidates
        candidates = []
        # zip the original batch_messages data with the responses
        for (unique_id, original_idx, _), response in zip(batch_messages, responses):
            # 2. PASS THE UNIQUE ID to response_to_individual
            # This ensures unique file creation for each candidate
            candidate = self.response_to_individual(response, unique_id)

            # 3. UPDATE with metadata. The original index is now retrieved from batch_messages
            failure_data = failures[original_idx]
            candidate.update({
                "id": f"{failure_data['individual']['id']}_cand{unique_id.split('_cand')[1]}",
                "parent_id": failure_data["individual"].get("parent_id"),
                "retry_reason": failure_data["error_msg"],
                "original_index": original_idx  # This is crucial for replacement later
            })
            candidates.append(candidate)

        return candidates

    def codebase_evaluate(self, codebase: list[dict]):
        max_candidates = 2  # Each failed individual generates 2 candidates

        # Phase 1: Parallel initial evaluation
        inner_runs = []
        failures = {}

        for idx, individual in enumerate(codebase):
            self.function_evals += 1
            individual.setdefault("id", f"{self.iteration}_{idx}_0")
            individual.setdefault("parent_id", None)

            if individual.get("exec_success", False):
                inner_runs.append(None)
                continue

            try:
                process = self._run_code(individual, idx)
                inner_runs.append(process)
            except Exception as e:
                logging.error(f"Initial execution failed for #{idx}: {str(e)}")
                failures[idx] = {"individual": self.mark_code_individual(individual, str(e)), "error_msg": str(e)}
                inner_runs.append(None)

        # ==================== MODIFICATION STARTS HERE ====================
        # Phase 2: Collect initial results
        for idx, process in enumerate(inner_runs):
            if process is None or idx in failures:
                continue

            individual = codebase[idx]
            try:
                # ... (communicate and read stdout_str as before) ...
                process.communicate(timeout=self.cfg.timeout)
                with open(individual["stdout_filepath"], 'r', encoding='utf-8', errors='replace') as f:
                    stdout_str = f.read()

                if traceback_msg := filter_traceback(stdout_str):
                    raise RuntimeError(traceback_msg)

                lines = [line.strip() for line in stdout_str.split('\n') if line.strip()]
                obj_value = float('inf')
                try:
                    obj_value = float(lines[-1]) if lines else float('inf')
                except (ValueError, IndexError):
                    try:
                        obj_value = float(lines[-2]) if len(lines) > 1 else float('inf')
                    except (ValueError, IndexError):
                        logging.warning(f"Could not parse objective from stdout for individual {idx}")

                individual.update({
                    "obj": obj_value if self.obj_type == "min" else -obj_value,
                    "raw_obj": obj_value,
                    "exec_success": True,
                    "curve": stdout_str
                })

            except subprocess.TimeoutExpired:
                # ... (handle timeout as before) ...
                process.kill()
                error_msg = f"Timeout after {self.cfg.timeout} seconds"
                failures[idx] = {"individual": self.mark_code_individual(individual, error_msg), "error_msg": error_msg}

            except Exception as e:
                # ... (handle other exceptions as before) ...
                error = self.extract_complete_error(e)
                error = self.strip_ansi(str(error))
                logging.error(f"Result processing failed for #{idx}: {error}")
                failures[idx] = {"individual": self.mark_code_individual(individual, error), "error_msg": error}
                with open(os.path.join(self.root_dir, 'thoughtbase/error_history.txt'), 'a',encoding='utf-8', errors='replace') as f:
                    f.write(error + '\n')

        if not failures:
            return codebase

        # Phase 3: Generate candidates (this part is already corrected and safe)
        candidates = self._generate_candidates(failures, max_candidates)

        # Phase 4: Parallel evaluation and collection for candidates
        candidate_runs = []
        candidate_map = {}
        for cand_idx, candidate in enumerate(candidates):
            try:
                process = self._run_code(candidate, candidate["response_id"])
                candidate_runs.append(process)
                candidate_map[len(candidate_runs) - 1] = cand_idx
            except Exception as e:
                logging.error(f"Candidate execution failed for ID {candidate['response_id']}: {str(e)}")
                candidate_runs.append(None)

        candidate_results = []
        for proc_idx, process in enumerate(candidate_runs):
            if process is None: continue
            cand_idx = candidate_map[proc_idx]
            candidate = candidates[cand_idx]
            try:
                # ... (communicate and read stdout_str for candidates) ...
                process.communicate(timeout=self.cfg.timeout)
                with open(candidate["stdout_filepath"], 'r', encoding='utf-8', errors='replace') as f:
                    stdout_str = f.read()

                if traceback_msg := filter_traceback(stdout_str):
                    raise RuntimeError(traceback_msg)

                lines = [line.strip() for line in stdout_str.split('\n') if line.strip()]
                obj_value = float('inf')
                try:
                    obj_value = float(lines[-1]) if lines else float('inf')
                except (ValueError, IndexError):
                    try:
                        obj_value = float(lines[-2]) if len(lines) > 1 else float('inf')
                    except (ValueError, IndexError):
                        logging.warning(
                            f"Could not parse objective from stdout for candidate {candidate['response_id']}")

                candidate.update({
                    "obj": obj_value if self.obj_type == "min" else -obj_value,
                    "raw_obj": obj_value,
                    "exec_success": True,
                    "curve": stdout_str
                })
                candidate_results.append({"original_index": candidate["original_index"], "individual": candidate})

            except Exception as e:
                # ... (handle candidate exceptions as before) ...
                error = self.extract_complete_error(e)
                error = self.strip_ansi(str(error))
                logging.error(f"Candidate processing failed for ID {candidate['response_id']}: {error}")
                with open(os.path.join(self.root_dir, 'thoughtbase/error_history.txt'), 'a', encoding='utf-8', errors='replace') as f:
                    f.write(str(error) + '\n')

        # Phase 5: Replace failures (this part is also safe)
        # ... (logic for replacement remains the same) ...
        replacement_count = 0
        for idx in failures:
            valid_candidates = [res["individual"] for res in candidate_results if
                                res["original_index"] == idx and res["individual"]["exec_success"] and not np.isinf(
                                    res["individual"]["obj"])]
            if valid_candidates:
                best_candidate = min(valid_candidates, key=lambda x: x["obj"])
                best_candidate["id"] = best_candidate["id"].split('_cand')[0]
                codebase[idx] = best_candidate
                replacement_count += 1
            else:
                codebase[idx].update({"obj": float('inf'), "raw_obj": float('inf'), "exec_success": False,
                                      "curve": f"No valid candidates found for failure: {failures[idx]['error_msg']}"})

        # **FINAL STEP**: After all parallel work is done, write all collected

        logging.info(f"Replaced {replacement_count}/{len(failures)} failed individuals.")
        return codebase

    def mark_code_individual(self, individual: dict, error_msg: str) -> dict:
        """Mark failed individuals"""
        return {
            **individual,
            "obj": float('inf') if self.obj_type == "min" else float('-inf'),
            "exec_success": False,
            "error": error_msg
        }

    def _run_code(self, code: dict, code_id):
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {code_id}")

        with open(self.output_file, 'w', encoding='utf-8', errors='replace') as file:
            file.writelines(code["code"] + '\n')

        with open(code["stdout_filepath"], 'w', encoding='utf-8', errors='replace') as f:
            eval_file_path = f'{self.root_dir}/evaluators/{self.problem}/main.py'
            process = subprocess.Popen(
                ['python', '-u', eval_file_path, self.root_dir, "train"],
                stdout=f,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace'  # 自动替换非法字符
            )

        block_until_running(code["stdout_filepath"], log_status=True,
                            iter_num=self.iteration, response_id=code_id)
        return process

    def save_codebase(self, codebase: list[dict]):
        """Save the current codebase to file organized by iteration"""
        # Create the base directory if it doesn't exist
        base_dir = os.path.join(self.root_dir, "codebase")
        os.makedirs(base_dir, exist_ok=True)

        # Create iteration-specific directory
        iter_dir = os.path.join(base_dir, f"iteration_{self.iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        # Save each individual's code and objective
        for idx, individual in enumerate(codebase):
            # Save code to a .py file
            code_filename = os.path.join(iter_dir, f"individual_{idx}_code.py")
            with open(code_filename, 'w', encoding='utf-8', errors='replace') as f:
                try:
                    f.write(individual["code"])
                except:
                    f.write('None')

            # Save objective to a .txt file
            obj_filename = os.path.join(iter_dir, f"individual_{idx}_obj.txt")
            with open(obj_filename, 'w', encoding='utf-8', errors='replace') as f:
                try:
                    f.write(str(individual["curve"]))
                except:
                    f.write(str(individual["obj"]))

            with open(os.path.join(self.root_dir, 'record.txt'), 'a', encoding='utf-8', errors='replace') as f:
                f.write(str(individual["obj"]) + '\n')

    def next_gen(self) -> None:
        # Calculate the average objective value of valid individuals.
        valid_objs = [
            ind["obj"] for ind in self.codebase
            if ind.get("exec_success", False) and not np.isinf(ind["obj"])
        ]

        if valid_objs:
            avg_obj = sum(valid_objs) / len(valid_objs)
            logging.info(f"Iteration {self.iteration}: Average objective = {avg_obj:.6f}")
            print(f"\n[Iter {self.iteration}] Avg Obj: {avg_obj:.6f}", flush=True)
        else:
            logging.warning(f"Iteration {self.iteration}: No valid objectives")
            print(f"\n[Iter {self.iteration}] No valid objectives", flush=True)

        # Filter individuals with objective values.
        codebase = [ind for ind in self.codebase if "obj" in ind]
        if not codebase:
            logging.error("No individuals with objectives in codebase!")
            return

        # Find the optimal solution for the current generation.
        objs = [individual["obj"] for individual in codebase]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        current_gen_best = codebase[best_sample_idx]

        # Record the optimal solution for the current generation.
        gen_best_dir = os.path.join(self.root_dir, "generation_best")
        os.makedirs(gen_best_dir, exist_ok=True)
        with open(os.path.join(gen_best_dir, f"gen_{self.iteration}_best_code.py"), 'w', encoding='utf-8', errors='replace') as f:
            f.write(current_gen_best["code"])
        with open(os.path.join(gen_best_dir, f"gen_{self.iteration}_best_obj.txt"), 'w', encoding='utf-8', errors='replace') as f:
            f.write(str(current_gen_best["obj"]))
        with open(os.path.join(gen_best_dir, f"gen_{self.iteration}_curve.txt"), 'w', encoding='utf-8', errors='replace') as f:
            f.write(current_gen_best["curve"])

        # Update the global optimal solution.
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = current_gen_best["code"]
            self.best_code_path_overall = current_gen_best["code_path"]

        # Update elite individuals.
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = current_gen_best
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")

        # Save the current code repository.
        self.save_codebase(codebase)

        # Save the global optimal solution.
        base_dir = os.path.join(self.root_dir, "best")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "best_code.py"), 'w', encoding='utf-8', errors='replace') as f:
            f.write(self.best_code_overall)
        with open(os.path.join(base_dir, "best_obj.txt"), 'w', encoding='utf-8', errors='replace') as f:
            f.write(self.elitist["curve"])

        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1

    def e_learning(self, elist):
        with open(os.path.join(self.root_dir, 'thoughtbase/thought_history.txt'), 'r', encoding='utf-8', errors='replace') as f:
            thoughts = f.read()
        with open(os.path.join(self.root_dir, 'thoughtbase/error_history.txt'), 'r', encoding='utf-8', errors='replace') as f:
            errors = f.read()
        user = e_learning_prompt(thoughts, errors, elist['code'], elist['curve'])
        messages = [{"role": "system", "content": self.e_learning_role}, {"role": "user", "content": user}]
        logging.info("E_learning prompt:")
        response = self.llm.multi_chat_completion([messages])[0]
        history_path = os.path.join(self.root_dir, 'thoughtbase/thought_history.txt')
        record_path = os.path.join(self.root_dir, 'thoughtbase/e_learning_record.txt')
        with open(record_path, 'a', encoding='utf-8', errors='replace') as f:
            f.write('\n' + response)
        user = metacognition_prompt(response, elist['code'], self.code_init, self.init_eval, self.problem_a)
        messages = [{"role": "system", "content": self.system_role}, {"role": "user", "content": user}]
        logging.info("Evolve start:")
        responses = self.llm.multi_chat_completion([messages], self.cfg.pop_size,
                                                   temperature=self.llm.temperature + 0.3)
        codebase = [self.response_to_individual(response, response_id) for response_id, response in
                    enumerate(responses)]
        codebase = self.codebase_evaluate(codebase)
        for individual in codebase:
            if individual['thought'] is not None:
                with open(history_path, 'a', encoding='utf-8', errors='replace') as f:
                    try:
                        f.write('\n' + individual['thought'] + '\nThe score:' + str(individual['obj']))
                    except:
                        f.write('\n' + individual['thought'] + '\nThe score:' + str(np.Inf))
        self.codebase = codebase
        self.next_gen()

    def Nature_Evo(self):
        pass

    def run(self):
        if all([not individual["exec_success"] for individual in self.codebase]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")

        # E-learning phase
        while self.function_evals < self.cfg.max_fe:
            self.e_learning(self.elitist)

        return self.best_code_overall, self.best_code_path_overall




import time
import hydra
import logging
import os
from pathlib import Path
import subprocess
from utils.utils import init_client
import shutil


def clear_record_history():
    thought_filename = 'record.txt'
    with open(thought_filename, 'w', encoding='utf-8') as f:
        pass


clear_record_history()


def clear_e_learning_history():
    thought_dir = "thoughtbase"
    thought_filename = 'e_learning_record.txt'
    os.makedirs(thought_dir, exist_ok=True)
    path = os.path.join(thought_dir, thought_filename)

    with open(path, 'w', encoding='utf-8') as f:
        pass


clear_e_learning_history()


def clear_thought_history():
    thought_dir = "thoughtbase"
    thought_filename = 'thought_history.txt'
    os.makedirs(thought_dir, exist_ok=True)
    path = os.path.join(thought_dir, thought_filename)

    with open(path, 'w', encoding='utf-8') as f:
        pass


clear_thought_history()


def clear_error_history():
    thought_dir = "thoughtbase"
    thought_filename = 'error_history.txt'
    os.makedirs(thought_dir, exist_ok=True)
    path = os.path.join(thought_dir, thought_filename)

    with open(path, 'w', encoding='utf-8') as f:
        pass


clear_error_history()
ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


def clear_codebase_directory(root_dir: str):
    codebase_dir = os.path.join(root_dir, "codebase")
    if not os.path.exists(codebase_dir):
        logging.info(f"Codebase directory {codebase_dir} doesn't exist, nothing to clear")
        return

    try:
        shutil.rmtree(codebase_dir)
        logging.info(f"Successfully cleared all contents of {codebase_dir}")
    except Exception as e:
        logging.error(f"Failed to clear codebase directory: {str(e)}")
        raise RuntimeError(f"Could not clear codebase directory: {str(e)}")

clear_codebase_directory(ROOT_DIR)

def clear_generation_directory(root_dir: str):
    codebase_dir = os.path.join(root_dir, "generation_best")
    if not os.path.exists(codebase_dir):
        logging.info(f"Codebase directory {codebase_dir} doesn't exist, nothing to clear")
        return
    try:
        shutil.rmtree(codebase_dir)
        logging.info(f"Successfully cleared all contents of {codebase_dir}")
    except Exception as e:
        logging.error(f"Failed to clear codebase directory: {str(e)}")
        raise RuntimeError(f"Could not clear codebase directory: {str(e)}")

clear_generation_directory(ROOT_DIR)

def clear_outputs_directory(root_dir: str):
    outputs_dir = os.path.join(root_dir, "outputs")
    if not os.path.exists(outputs_dir):
        logging.info(f"Codebase directory {outputs_dir} doesn't exist, nothing to clear")
        return

    try:
        shutil.rmtree(outputs_dir)
        logging.info(f"Successfully cleared all contents of {outputs_dir}")
    except Exception as e:
        logging.error(f"Failed to clear codebase directory: {str(e)}")
        raise RuntimeError(f"Could not clear codebase directory: {str(e)}")

clear_outputs_directory(ROOT_DIR)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)

    from ELE import ELE

    lhh = ELE(cfg, ROOT_DIR, client)

    start = time.time()
    best_code_overall, best_code_path_overall = lhh.run()
    end = time.time()

    best_path = best_code_path_overall.replace(".py", ".txt").replace("code", "response")
    logging.info(f"Best Code Path Overall: {best_path, best_code_path_overall}, time:{end-start}")


if __name__ == "__main__":
    main()

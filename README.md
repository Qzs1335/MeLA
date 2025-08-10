Welcome to MeLA!

Usage Instructions
1. PyCharm Setup
•
Ensure PyCharm is installed and configured.

2. Script Parameters Configuration
In main.py, set the following ​Script Parameters​ (example for the ​ACS​ problem):

llm_client=openai  
llm_client.api_key="Your_API_Key"  
problem=acs  
init_pop_size=20  
pop_size=10  
max_fe=50  
timeout=300

README
Usage Instructions
1. PyCharm Setup
•
Ensure PyCharm is installed and configured.

2. Script Parameters Configuration
In main.py, set the following ​Script Parameters​ (example for the ​ACS​ problem):

llm_client=openai  
llm_client.api_key="Your_API_Key"  
problem=acs  
init_pop_size=20  
pop_size=10  
max_fe=50  
timeout=300

3. Adding a New Problem
When creating a new problem:

•
Follow the structure of existing problems in the evaluatorsfolder (e.g., the ​ACS​ example).

•
Add a corresponding ​YAML​ configuration file in cfg/problemwith the following format:

problem_name: acs  
problem_type: scso  
obj_type: min  
func_name: heuristics

4. Modifying ELE.py
For different problems, update these two lines in ELE.py:

file_path = os.path.join(current_dir, 'experts', '<problem_type>.txt')  
file_path = os.path.join(current_dir, 'promoter', '<init_code_file>.txt')

## Required .txt Files for Each Problem

| Problem    | Experts File (`experts/`) | Promoter File (`promoter/`) |
|------------|--------------------------|----------------------------|
| ACS, WSN   | `scso.txt`               | `init_code_empty.txt`      |
| TSP        | `tsp.txt`                | `init_code_tsp.txt`        |
| BPP        | `bpp.txt`                | `init_code_bpp.txt`        |

5. Running the Algorithm
Execute main.pyin PyCharm with the configured parameters.

This READMEprovides a concise translation of the original instructions. Let me know if further adjustments are needed! 🚀

def problem_role():
    prompt = f'''Act as a NP-hard problem analyst. 
Please analyze the characteristics and solutions of the NP-hard problem.
'''
    return prompt

def system_role():
    prompt = f'''Act as an expert evolutionary computing.  
Please iteratively improve the provided algorithm.  
The code generation format must strictly follow the example below:
    {{thought process:
    1.xxx
    2.xxx
    ...}}     
    ```python
    import numpy as np 
    def heuristics_v1(* The same as the sample code input *):
        * The rest remains unchanged. *
        #EVOLVE-START
        * Your optimized code *
        #EVOLVE-END       
        return Positions
    ```  
Do not import other libraries 
    '''
    return prompt

def error_role():
    prompt = '''You are an elite Code Debugging.
Please correct the python code.
The code generation format must strictly follow the example below:
    {thought process:
    1.xxx
    2.xxx
    ...}   
    ```python
    import numpy as np 
    def heuristics_v1(* The same as the sample code input *):
        * The rest remains unchanged. *
        #EVOLVE-START
        * Your optimized code *
        #EVOLVE-END       
        return Positions
    ```  
    '''
    return prompt

def e_learning_role():
    prompt = f'''You are another version of yourself, a process of thinking and a set of errors through which you come to understand yourself.
Please analyze the thought process records, errors and the optimal algorithm. 
'''
    return prompt

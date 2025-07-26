def problem_prompt(problem):
    prompt = f'''I have an existing problem with the code as follows:{problem}.
Please analyze the problem.
Your analysis must be within 50 words.
'''
    return prompt

def return_promoter_init(init_code,init_eval,code,problem):
    prompt = f'''{problem}
I have an initial algorithm with the code as follows:{init_code}.
And the optimization history it presents in the problem is as follows:{init_eval}.
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1.Analyze the history of fitness values and optimize the algorithm with the goal of surpassing the optimal value.
2.You will notice that there are #EVOLVE-START and #EVOLVE-END comments in the following code. The code within these comments is the part that you need to optimize.
3.The code:{code}. Analyze the algorithm, optimize the algorithm. Record your thought process in the {{}} brackets. 
4.Your thought process must be within 50 words.
'''
    return prompt

def error_prompt(str_error, code_str):
    prompt = f'''The code you generated {code_str} has the following error {str_error}.
Please make it correct and functional.
'''
    return prompt

def e_learning_prompt(thoughts, errors, code, history):
    prompt = f'''The thinking process and score of each algorithm are as follows:{thoughts}.
The errors are as follows:{errors}.
You should avoid the errors and ensure that no new error.
The optimal algorithm is as follows:{code}.
Please conduct metacognitive reflection on your own thinking process, scores and mistakes.
1.Analyze the important considerations for optimizing fitness values.
2.The excellent components that should be retained in the optimal algorithm.
3.The components with better performance that need to be retained.
4.Your output content must be within 80 words.
'''
    return prompt

def metacognition_prompt(metacognition,code,init_code,init_eval,problem):
    prompt = f'''The reflection results of metacognition are as follows:{metacognition}.
I have a existing algorithm with their codes as follows:{init_code}.
The optimization history it presents in the optimization problem is as follows:{init_eval}.
Please retain the advantageous components and innovate to improve the deficient ones.
1.You will notice that there are #EVOLVE-START and #EVOLVE-END comments in the following code. The code within these comments is the part that you need to optimize.
2.The code:{code}. Analyze the algorithm, optimize the algorithm.Record your thought process in the {{}} brackets. 
3.Your thought process must be within 50 words.
'''
    return prompt



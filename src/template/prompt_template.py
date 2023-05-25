
# Prompt template for the generating instruction
INSTRUCTION_PROMPT = "Given the existing instructions, please generate a list of diverse python code editing instructions. The new instructions should address diverse editing task. Please ensure that the instructions are clear and diverse. Include any relevant variable name in the instruction."
INSTRUCTION_PROMPT_WITH_TYPE = "Given the existing instructions, please generate a list of diverse python code editing instructions. The new instructions should address diverse editing tasks related to `{edit_task}`. Please ensure that the instructions are clear and diverse. Include any relevant variable name in the instruction."
EDIT_TYPES = [
        "Completing incomplete code",
        "Optimizing code time complexity or space(memory) complexity",
        "Fixing bugs",
        "Adding comments or docstring",
        "Create unit tests",
        "Refactoring code",
        "Improving code readability",
        "Implementing error handling",
        "Enhancing performance",
        "Updating library dependencies",
        "Adapting code for accessibility",
        "Adhering to coding standards",
        "Ensuring code security",
        "Modularizing code",
        "Migrating to a new framework",
        "Adding logging and monitoring",
        "Implementing code linting",
        "Internationalizing code",
        "Removing dead code",
        "Implementing caching mechanisms",
        "Applying design patterns",
        "Addressing memory leaks",
        "Implementing multithreading",
        "Reducing code duplication",
        "Integrating APIs and services",
        "Adding new features/functionality",
]

# Prompt template for the generating scenario
SCENARIO_PROMPT = """Given a python code editing task, please come up with 10 diverse scenarios concise description where this python code editing task could be performed or come from.

{SHOT1}


{SHOT2}
"""


# Prompt template for the generating input/output code pair
INSTANCE_PROMPT = "Given python code editing task instructions and their scenarios where the task instruction could be used, you need to come up with examples for the following code editing tasks.  You need to generate input and output code pair and make sure your variable names are suitable for the scenario. The input code is related to the task instruction, but must NOT meet the task requirements. The output code fulfills the task requirements based on input code.\n\n"
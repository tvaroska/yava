"""

    Singleton prompts - repo for all prompts in the module

"""
from pathlib import Path
from promptgit.repo import PromptRepo

PROMPTS_DIR = 'prompts'

current_directory = Path.cwd()

if (current_directory / PROMPTS_DIR).exists():
    prompts_location = str(current_directory / PROMPTS_DIR)
# Check one dir up
elif (current_directory.parents[0] / PROMPTS_DIR).exists():
    prompts_location = str(current_directory.parents[0] / PROMPTS_DIR)
else:
    raise FileNotFoundError(PROMPTS_DIR)

prompts = PromptRepo(prompts_location)

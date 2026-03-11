#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import copy

from .utils import extract_code_blocks, remove_duplicate_code, count_tokens_for_openai, get_avoid_words
from .transit_func_utils import experiences2text
from .evaluator import evaluate_transit_code
from ..env_info import DocString, TransitCodeExample

def _is_text_based_state(experiences):
    """Detect if experiences use TextState (text-based) vs object-centric states."""
    if not experiences:
        return False
    # Check the first experience's input_state
    state = experiences[0].input_state
    # TextState has .observation attribute, object-centric has .objs
    if hasattr(state, 'observation') and hasattr(state, 'available_actions'):
        return True
    # Or if it has get_objs_by_obj_type method, it's object-centric
    if hasattr(state, 'get_objs_by_obj_type'):
        return False
    # Check if it's a simple string (also text-based)
    if isinstance(state, str):
        return True
    return False

def get_text_based_docstring():
    """Get documentation for text-based environments."""
    return '''
"""
class TextState:
    """For text-based environments (maze, wordle, text adventures, etc.)"""
    Attributes:
        observation (str): The text observation containing all state information
        available_actions (tuple): Tuple of available actions as strings

    Methods:
        deepcopy(): Returns a copy of the state
        get_str_w_ints_w_touching(): Returns string representation
        
    IMPORTANT: You must PARSE the observation text using string operations or regex.
    Do NOT use object methods like get_objs_by_obj_type() - they don't exist for TextState!
    
    Example state.observation:
    "Your goal: go to a red box\\nIn front of you...\\nAvailable actions: [\\"turn left\\", ...]"
"""
'''

def get_text_based_example():
    """Get transition code example for text-based environments."""
    return '''
def transition(state, event):
    """
    Args:
        state (TextState): State with .observation (str) and .available_actions (tuple)
        event (str): action taken in the input state
    Returns:
        new_state (TextState): New state with updated observation
    """
    import copy
    
    # IMPORTANT: For text-based environments, parse the .observation string
    # and create a new TextState with the updated text.
    
    # Copy the state (don't modify in place)
    new_state = state.deepcopy()
    
    # Parse the observation text to extract current information
    obs = new_state.observation
    
    # Example: Parse position from text like "position is at position 1, 2"
    import re
    match = re.search(r'position (\\d+), (\\d+)', obs)
    if match:
        x, y = int(match.group(1)), int(match.group(2))
        # Update position based on action
        if event == 'move right':
            y += 1
        elif event == 'move left':
            y -= 1
        elif event == 'move down':
            x += 1
        elif event == 'move up':
            x -= 1
        # Generate new observation with updated position
        new_obs = re.sub(r'position \\d+, \\d+', f'position {x}, {y}', obs)
        new_state.observation = new_obs
    
    return new_state
'''

def init_transit(experiences, llm, verbose=True,):
    verbose_flag = verbose

    # Detect if we're using text-based states
    is_text_based = _is_text_based_state(experiences)
    
    text_experiences = experiences2text(experiences)
    _experiences = copy.deepcopy(experiences)
    while count_tokens_for_openai(text_experiences) > 5120 and len(_experiences) > 1:
        _experiences = _experiences[:-1]
        text_experiences = experiences2text(_experiences)

    # Choose the appropriate system message and doc/example based on state type
    if is_text_based:
        system_message = TEXT_FIRST_SYSTEM_MESSAGE
        doc_string = get_text_based_docstring()
        code_example = get_text_based_example()
        # Update the experiences text to emphasize text parsing
        text_experiences = f"""**IMPORTANT: This is a TEXT-BASED environment!**
The state has .observation (str) and .available_actions (tuple) attributes.
You must PARSE the observation text, not use object methods!

{text_experiences}"""
    else:
        system_message = FIRST_SYSTEM_MESSAGE
        doc_string = DocString
        code_example = TransitCodeExample

    chat_history = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': FIRST_MESSAGE.format(
            experiences=text_experiences,
            DocString=doc_string,
            TransitCodeExample=code_example,
        ),},
    ]
    if verbose_flag:
        print('-'*20 + 'Guessing initial code: Prompts' + '-'*20)
        for chat in chat_history:
            print()
            print(chat['role'] + ':')
            print(chat['content'])
            print()

    llm_model_args = {'logit_bias': get_avoid_words(['class',])}
    with llm.track() as cb:
        with llm.track_new() as new_cb:
            gen = llm(chat_history, model_args=llm_model_args,)
            gen = gen.choices[0].message
    if verbose_flag:
        print('*'*20 + 'Guessing Initial code: Machine Reply' + '*'*20)
        print(gen.content)
        print(cb)

    code_blocks = extract_code_blocks(gen.content)
    while True:
        code = '\n'.join(code_blocks)
        code = remove_duplicate_code(code)
        result = evaluate_transit_code(code, experiences,)
        if result['compilation_error'] is not None and len(code_blocks) > 1:
            if verbose_flag:
                print('Compilation Error:', result['compilation_error'])
            code_blocks = code_blocks[:-1]
        else:
            break
    if verbose_flag:
        print('\nResults:', {
            k: v for k, v in result.items()
            if len(str(v)) < 100
        })

    success_flag = result['success_flag']
    chat_history.append({'role': 'assistant', 'content': gen.content})
    final_outputs = {
        'chat_history': chat_history,
        'result': result,
        'configurations': {
            'experiences': experiences,
            'filename': osp.abspath(__file__),
        },
        'costs': {k:v for k,v in cb.usage.items() if k != '_lock'},
        'new_costs': {k:v for k,v in new_cb.usage.items() if k != '_lock'},
        'code': code,
    }
    return {
        'success_flag': success_flag,
        'final_outputs': final_outputs,
        'code': code,
    }

FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in an environment. Your goal is to model the logic of the world in python. You will be provided experiences in the format of (state, action, next_state) tuples.

**CRITICAL INSTRUCTION - READ CAREFULLY:**
The state can be in one of two formats:

1. **OBJECT-CENTRIC**: Contains physical objects with positions/velocities
   - Use methods like `state.get_objs_by_obj_type("player")`
   - Modify object attributes like `player.velocity_x`

2. **TEXT-BASED**: Contains text descriptions (for maze, wordle, text adventures)
   - You MUST parse the text using string operations or regex
   - You MUST generate new text for the next state
   - Example: Extract "position 1, 2" from text and return updated text

**HOW TO IDENTIFY:**
- If states look like: `{player: (x=1, y=2), ball: (x=3, y=4)}` → Object-centric
- If states look like: "Your current position is at position 1, 2. There are walls above you." → Text-based

**DO NOT** use ObjList methods for text-based states. Parse the text directly!

You need to implement the python code to model the logic of the world, as seen in the provided experiences. Please follow the template to implement the code. The code needs to be directly runnable on the state and return the next state in python as provided in the experiences.
'''.strip()

TEXT_FIRST_SYSTEM_MESSAGE = '''
You are a robot exploring in a TEXT-BASED environment. Your goal is to model the logic of the world in python.

**CRITICAL: THIS IS A TEXT-BASED ENVIRONMENT**

The state is a TextState object with these attributes:
- `state.observation` (str): The text observation containing all state information
- `state.available_actions` (tuple): Available actions as strings

**YOU MUST:**
1. Parse the `state.observation` text using string operations or regex
2. Generate new text for the output state's observation
3. Return a TextState with the updated observation

**DO NOT:**
- Use `state.get_objs_by_obj_type()` - TextState doesn't have this method!
- Try to access `.objs` or `.velocity_x` - these don't exist!
- Modify object attributes directly

**EXAMPLE:**
```python
def transition(state, event):
    import copy, re
    new_state = state.deepcopy()  # Copy the state
    obs = new_state.observation
    
    # Parse text to extract information
    match = re.search(r'position (\\d+), (\\d+)', obs)
    if match:
        x, y = int(match.group(1)), int(match.group(2))
        # Update based on action
        if event == 'move right':
            y += 1
        # Generate new observation text
        new_obs = re.sub(r'position \\d+, \\d+', f'position {x}, {y}', obs)
        new_state.observation = new_obs
    
    return new_state
```

You need to implement the python code to model the logic of the world, as seen in the provided experiences. The code needs to be directly runnable on the state and return the next state in python as provided in the experiences.
'''.strip()

FIRST_MESSAGE = '''
You need to implement python code to model the logic of the world as seen in the following experiences:

{experiences}

Please implement code to model the logic of the world as demonstrated by the experiences. Please implement the transition function following the template. The code needs to be directly runnable on the inputs of (state, action) and return the next state in python as provided in the experiences. You should only change the velocities of objects in the state. Please do not use any class such as ObjList to initialize the state. You can directly assign `new_state=state` to copy the state.

{DocString}

{TransitCodeExample}

Please implement code to model the logic of the world as demonstrated by the experiences. Please implement the code following the template. Feel free to implement the helper functions you need. You can also implement the logic for difference actions in different helper functions. However, you must implement the ` transition ` function as the main function to be called by the environment. The code needs to be directly runnable on the inputs as (state, action) and return the next state in python as provided in the experiences. Let's think step by step.
'''.strip()

#!/usr/bin/env python
# coding=utf-8

DocString = '''
Here are some documentation explaining the API for the environment.

IMPORTANT: The state can be either OBJECT-CENTRIC or TEXT-BASED depending on the environment:

1. For OBJECT-CENTRIC environments (e.g., atari-like games with physical objects):
    The state is an ObjList containing objects with positions and velocities.

2. For TEXT-BASED environments (e.g., maze, wordle, text adventures):
    The state is a text string containing all relevant information.
    You must PARSE the text to extract information and GENERATE text for the next state.

"""
class StateTransitionTriplet:
    Attributes:
        input_state (ObjList or str): input state - either list of objects or text
        event (str): action taken in the input state
        output_state (ObjList or str): output state - either list of objects or text

class ObjList:
    """For object-centric environments only"""
    Attributes:
        objs (list of Obj)

    Methods:
        get_objs_by_obj_type(obj_type: str) -> list[Obj]:
            Returns list of objects with the input obj_type

        create_object(obj_type: str, x: int, y: int) -> ObjList:
            Returns a new instance of ObjList with the new object (with obj_type, x, y) added

class Obj:
    """For object-centric environments only"""
    Attributes:
        id (int): id of the object
        obj_type (string): type of the object
        velocity_x (int): x-axis velocity of the object
        velocity_y (int): y-axis velocity of the object

    Methods:
        touches(obj: Obj, touch_side: int, touch_percent: float) -> bool:
            Returns whether this Obj is touching the input obj (True/False)
            based on the input touch_side (0 = left, 1 = right, 2 = up, 3 = down) and touch_percent (threshold for touching area percentage)

class TextState:
    """For text-based environments (maze, wordle, etc.)"""
    Attributes:
        observation (str): The text observation containing all state information

    Methods:
        Extract information by parsing the text (e.g., using string operations, regex)
        Generate next state by constructing appropriate text output
"""
'''

TransitCodeExample = '''
Here are examples of transition functions depending on the environment type:

## Example 1: Object-Centric Environment (e.g., games with physical objects)
```
def transition(state, event):
    """
    Args:
        state (ObjList): list of objects in the input state
        event (str): action taken in the input state
    Returns:
        new_state (ObjList): list of objects in the output state
    """
    # Example of transition, just showing how to access objects and modify them
    new_state = state
    player = state.get_objs_by_obj_type("player")[0]
    ball = state.get_objs_by_obj_type("ball")[0]
    if player.touches(ball, 0, 1.0):
        ball.velocity_x = -ball.velocity_x
    if event == 'UP':
        player.velocity_y = player.velocity_y + 1
    return new_state
```

## Example 2: Text-Based Environment (e.g., maze, wordle, text adventures)
```
def transition(state, event):
    """
    Args:
        state (str): text observation containing all state information
        event (str): action taken in the input state
    Returns:
        new_state (str): text observation for the next state
    """
    # IMPORTANT: For text-based environments, you must PARSE the input text
    # to extract state information, then GENERATE the output text.
    #
    # Example for a maze: Extract position from "Your current position is at position X, Y"
    # and return updated text "Your current position is at position X', Y'"
    #
    # Example for wordle: Compute feedback based on guess vs target word
    # and return feedback like "y b g b y"
    #
    # Parse the state to extract current information
    import re
    # Example: Extract position from text like "position is at position 1, 2"
    match = re.search(r'position (\d+), (\d+)', state)
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
        # Generate new state text with updated position
        new_state = re.sub(r'position (\d+), (\d+)', f'position {x}, {y}', state)
        return new_state
    return state
```

**CRITICAL**: Analyze the experiences provided. If states contain text like:
- "Your current position is at position X, Y"
- "The goal is at position 8, 6"
- "There are walls to your left"
Then you are in a TEXT-BASED environment and MUST parse/generate text, not use ObjList methods!
'''

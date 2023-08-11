"""
Copyright 2023 by Raytheon BBN Technologies All Rights Reserved

# Utility functions used to operate on yaml objects
# Exclusively used by client.py (right now)
# Relies on data from MAGIC_skills.py
"""
import time
import yaml

#temporary for testing/demo purposes
import random


def _get_generic_yaml():
    # Read generic file into yaml object
    with open("demo1.yaml", 'r') as f:
        return yaml.safe_load(f)

def _get_skill_steps(skill):
    with open("skills.yaml", 'r') as f:
        skills = yaml.safe_load(f)
    return skills['skills'][skill]['steps']


class Message(dict):
    def __init__(self, skill, errors=False) -> None:
        # Get the basic yaml
        super().__init__(_get_generic_yaml())
        self._initialize_skill(skill)
        # enable message features
        if errors:
            self['current errors']['populated'] = True
        self.update_time()

    def _initialize_skill(self, skill):
        # Get the skill steps we need
        steps = _get_skill_steps(skill)
        # Get current actions (holds steps)
        user_current_actions = self['users current actions right now']
        # Set current skill field
        user_current_actions['current skill']['number'] = skill
        # Place new steps into the yaml
        user_current_actions['steps'] = [
            {'number': i+1, 'name': step, 'state': 'unobserved' if i else 'current', 'confidence': 1}
            for i, step in enumerate(steps)
        ]

    def update_step(self, step_id):
        steps = self['users current actions right now']['steps']
        previous_step_id = next(
            (i for i, s in enumerate(steps) if s['state'] == 'current'), None)
        if previous_step_id == step_id:
            return 
        else:
            # naive step completion
            for i in range(step_id):
                steps[i]['state'] = 'done'
            steps[step_id]['state'] = 'current'
            for i in range(step_id+1, len(steps)):
                steps[i]['state'] = 'unobserved'

    def update_steps_state(self, steps):
        steps = self['users current actions right now']['steps'] = steps

    def update_errors(self, error_desc):
        self['current errors']['errors'] = [error_desc] if error_desc else []

    def update_time(self, t=None):
        self['header']['transmit timestamp'] = t or time.time()

    def __str__(self):
        return yaml.dump(dict(self))


def _get_initial_message(skill, errors=False):
    # Get the basic yaml
    message = _get_generic_yaml()
    # Get the skill steps we need
    steps = _get_skill_steps(skill)
    # Update yaml with relevant steps
    message = _initialize_skill(message, skill, steps)
    # enable message features
    if errors:
        message['current errors']['populated'] = True
    # Update timestamp
    message = _update_timestamp(message)
    return message


def _initialize_skill(message, skill, steps):
    '''Places provided steps into provided yaml object'''
    # Get current actions (holds steps)
    user_current_actions = message['users current actions right now']
    # Set current skill field
    user_current_actions['current skill']['number'] = skill
    # Place new steps into the yaml
    user_current_actions['steps'] = [
        {'number': i+1, 'name': step, 'state': 'unobserved' if i else 'current', 'confidence': 1}
        for i, step in enumerate(steps)
    ]
    # update timestamp
    message = _update_timestamp(message)
    return message
    
def _update_timestamp(message): 
    # update timestamp
    message['header']['transmit timestamp'] = time.time()
    return message


def update_yaml_message(message): 
    yaml_steps = message['users current actions right now']['steps']
    
    next_step_is_current = False
    
    # Iterate through the steps and update
    for step in range(len(yaml_steps)):
        current_step = yaml_steps[step]
        if current_step['state'] == 'current':
            # The old 'current' step is now complete
            current_step['state'] = 'done'
            next_step_is_current = True #the next step is the new 'current' step      
        elif next_step_is_current:
            # Update step to be the current step
            current_step['state'] = 'current'
            next_step_is_current = False
            
        # Place updated step into steps list
        yaml_steps[step] = current_step

    # Place updated steps into yaml and convert back to a string
    message['users current actions right now']['steps'] = yaml_steps
    # Update timestamp
    message = _update_timestamp(message)
    return message

import os
import re
import time
import orjson
import asyncio

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import zmq.asyncio
import yaml
import yaml_messages

import ptgctl
import ptgctl.util


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


class MsgSession:
    def __init__(self, skill_id):
        self.steps = {}
        self.message = yaml_messages.Message(skill_id, errors=True)

    def on_reasoning_step(self, data):   
        if 'all_steps' not in data:
            return
        self.message.update_steps_state(data['all_steps'])
        #self.message.update_step(data['step_id'])
        self.message.update_errors(data['error_description'] if data['error_status'] else False)
        return str(self.message)



class ZMQClient:
    def __init__(self, address):
        assert address
        self.address = address
        self.context = zmq.asyncio.Context()

    async def __aenter__(self):
        # Connect to the server
        print("Connecting to server…")
        self.socket = socket = self.context.socket(zmq.REQ)
        socket.connect(self.address)
        print("Connected...", self.address)
        return self

    async def __aexit__(self, *a):
        self.socket.close()

    async def send(self, message):
        # Send the yaml string to the server, check for proper response
        await self.socket.send_string(message)
        response = await self.socket.recv_string()
        if "Error" in response:
            raise IOError(f"ERROR: Invalid response from server - {response}")


class RecipeExit(Exception):
    pass

class InvalidMessage(Exception):
    pass


class MsgApp:
    ORG_NAME = 'nyu'
    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'bbn_msgs',
                              password=os.getenv('API_PASS') or 'bbn_msgs')
        
    @ptgctl.util.async2sync
    async def run_ctl_listener(self, address=os.getenv("ZMQ_ADDRESS")):
        name = self.ORG_NAME
        # this loop should be modified and incorporated into client code
        # so it can listen to and respond to the server
        # Connect to the server
        context = zmq.Context()
        print(f"Connecting to server [{address}]…")
        socket = context.socket(zmq.DEALER)
        socket.connect(address)
        try:
            socket.send_string(f"{self.ORG_NAME}:OK")
            print("Connected...")
            while True:
                if socket.poll(timeout=2):
                    response = socket.recv_string()
                    print(f"{name}: Received message: {response}")
                    try:                    
                        out_msg = self.handle_control_message(response)
                    except Exception as e:
                        out_msg = str(e)
                        print("Error:", out_msg)
                    msg = f"{name}:{out_msg or 'OK'}"
                    print('msg:', msg)
                    socket.send_string(msg, flags=zmq.DONTWAIT)
                time.sleep(.5)
        finally:
            socket.close()

    name_translate = {'-': None, '?': None, '.': None}
    cmd_translate = {'stopped': 'stop', 'started': 'start', 'done': 'stop'}
    recipe_translate = {'m2': 'tourniquet'} # FIXME: !! this should be stored in the DB
    def handle_control_message(self, msg):
        match = re.search(r'(\w+) (\w+) (\w+)', msg.lower())
        group, name, verb = 'experiment', None, msg
        if match:
            group = match.group(1)
            name = match.group(2)
            verb = match.group(3)
            name = self.name_translate.get(name, name)
            verb = self.cmd_translate.get(verb, verb)
        if group == 'experiment':
            if verb == 'start':
                return
            if verb == 'stop':
                self.api.session.stop_recipe()
                return
            if verb == 'pause':
                return
        if group == 'skill':
            if verb == 'start':
                assert name, f"start what?"
                #if name not in self.recipe_translate:
                #    raise InvalidMessage(f"Unsupported skill {name}")
                name = self.recipe_translate.get(name, name)
                self.api.session.start_recipe(name)
                return
            if verb == 'stop':
                self.api.session.stop_recipe()
                return
            if verb == 'pause':
                return
        if group == 'record':
            if verb == 'start':
                self.api.recordings.start(name)
                return
            if verb == 'stop':
                self.api.recordings.stop()
                return
        
        raise InvalidMessage(f"Unrecognized message: {msg}")


    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        '''Persistent running app, with error handling and recipe status watching.'''
        while True:
            try:
                recipe_id = self.api.session.current_recipe()
                while not recipe_id:
                    print("waiting for recipe to be activated")
                    recipe_id = await self._watch_recipe_id(recipe_id)
                
                print("Starting recipe:", recipe_id)
                await self.run_recipe(recipe_id, *a, **kw)
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_recipe_id(self, recipe_id):
        async with self.api.data_pull_connect('event:recipe:id') as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    print(sid, ts, data, recipe_id)
                    if data != recipe_id:
                        return data

    def start_session(self, skill_id, prefix=None):
        '''Initialize the action session for a recipe.'''
        if not skill_id:
            raise RecipeExit("no skill set.")
        skill = self.api.recipes.get(skill_id) or {}
        skill_id = skill.get('skill_id')
        if not skill_id:
            raise RecipeExit("skill has no skill_id key.")
        self.session = MsgSession(skill_id)

    async def run_recipe(self, recipe_id, address=os.getenv("ZMQ_ADDRESS"), prefix=None):
        '''Run the recipe.'''
        self.start_session(recipe_id, prefix=prefix)

        # stream ids
        reasoning_sid = f'{prefix or ""}reasoning:check_status'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([reasoning_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, ZMQClient(address) as zq:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            self.start_session(recipe_id, prefix=prefix)
                            continue
                        if self.session is None:
                            continue

                        # predict actions
                        preds = None
                        if sid == reasoning_sid:
                            preds = self.session.on_reasoning_step(orjson.loads(d))
                        if preds:
                            await zq.send(preds)
                        


if __name__ == '__main__':
    import fire
    fire.Fire(MsgApp)

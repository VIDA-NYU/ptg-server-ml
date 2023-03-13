import os
import orjson
import asyncio
import logging
import ptgctl
import ptgctl.util
from os.path import join
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
#ptgctl.log.setLevel('WARNING')



RECIPE_SID = 'event:recipe:id'
SESSION_SID = 'event:session:id'
UPDATE_STEP_SID = 'event:recipe:step'
ACTIONS_CLIP_SID = 'clip:action:steps'
ACTIONS_EGOVLP_SID = 'egovlp:action:steps'
OBJECTS_SID = 'detic:image:v2'
REASONING_STATUS_SID = 'reasoning:check_status'
REASONING_ENTITIES_SID = 'reasoning:entities'


CONFIGS = {'tagger_model_path': join(os.environ['REASONING_MODELS_PATH'], 'recipe_tagger'),
           'bert_classifier_path': join(os.environ['REASONING_MODELS_PATH'], 'bert_classifier')}


#def data_pull_connect(self, stream_id: str, ack=True, **kw):
#    if isinstance(stream_id, (list, tuple)):
#        stream_id = '+'.join(stream_id)
#    if '+' in stream_id or stream_id == '*':
#        kw.setdefault('batch', True)
#    return self._ws('data', stream_id, 'pull?ack=True', cls=ptgctl.core.DataStream, ack=True, **kw)
#ptgctl.API.data_pull_connect = data_pull_connect

class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

        self.state_manager = StateManager(CONFIGS)

    def start_recipe(self, recipe_id):
        logger.info(f'Starting recipe, ID={str(recipe_id)}')
        if recipe_id is not None:
            recipe = self.api.recipes.get(recipe_id)
            logger.info(f'Loaded recipe: {str(recipe)}')
            step_data = self.state_manager.start_recipe(recipe)
            logger.info(f'First step: {str(step_data)}')

            return step_data

    async def run_reasoning(self, prefix='', top=5, use_egovlp=True):
        actions_sid = prefix + ACTIONS_EGOVLP_SID if use_egovlp else prefix + ACTIONS_CLIP_SID
        objects_sid = prefix + OBJECTS_SID
        re_check_status_sid = prefix + REASONING_STATUS_SID
        re_entities_sid = prefix + REASONING_ENTITIES_SID

        async with self.api.data_pull_connect([actions_sid, objects_sid, RECIPE_SID, SESSION_SID, UPDATE_STEP_SID], ack=True) as ws_pull, \
                   self.api.data_push_connect([re_check_status_sid, re_entities_sid], batch=True) as ws_push:

            recipe_id = self.api.session.current_recipe()
            first_step = self.start_recipe(recipe_id)
            if first_step is not None:
                await ws_push.send_data([orjson.dumps(first_step)], re_check_status_sid)

            entities = self.state_manager.get_entities()
            if entities is not None:
                logger.info(f'Sending entities for all steps: {str(entities)}')
                await ws_push.send_data([orjson.dumps(entities)], re_entities_sid)

            detected_actions = None
            detected_objects = None

            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    if sid == RECIPE_SID:  # A call to start a new recipe
                        recipe_id = data.decode('utf-8')
                        first_step = self.start_recipe(recipe_id)
                        if first_step is not None:
                            await ws_push.send_data([orjson.dumps(first_step)], re_check_status_sid)
                        entities = self.state_manager.get_entities()
                        if entities is not None:
                            logger.info(f'Sending entities for all steps: {str(entities)}')
                            await ws_push.send_data([orjson.dumps(entities)], re_entities_sid)
                        continue

                    elif sid == UPDATE_STEP_SID:  # A call to update the step
                        step_index = int(data)
                        updated_step = self.state_manager.set_user_feedback(step_index)
                        if updated_step is not None:
                            await ws_push.send_data([orjson.dumps(updated_step)], re_check_status_sid)
                        continue

                    elif sid == SESSION_SID:  # A call to start a new session
                        #self.state_manager.reset()
                        continue

                    elif sid == objects_sid:  # A call sending detected objects and bounding boxes
                        detected_objects = orjson.loads(data)
                        logger.info(f'Perception objects: {str(detected_objects)}')

                    elif sid == actions_sid:  # A call sending detected actions
                        detected_actions = orjson.loads(data)
                        detected_actions = sorted(detected_actions.items(), key=lambda x: x[1], reverse=True)[:top]
                        logger.info(f'Perception actions: {str(detected_actions)}')

                    if detected_objects is not None and detected_actions is not None:
                        recipe_status = self.state_manager.check_status(detected_actions, detected_objects)
                        logger.info(f'Reasoning outputs: {str(recipe_status)}')
                        if recipe_status is not None:
                            await ws_push.send_data([orjson.dumps(recipe_status)], re_check_status_sid)
                            # Reset the values of the detected inputs
                            detected_actions = None
                            detected_objects = None

    @ptgctl.util.async2sync
    async def run(self, *args, **kwargs):
        while True:
            try:
                recipe = self.api.session.current_recipe()
                if recipe:
                    await self.run_reasoning(*args, **kwargs)
                else:
                    print(recipe)
                    await asyncio.sleep(1)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3)


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)

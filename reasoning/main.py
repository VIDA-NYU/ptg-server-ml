import os
import orjson
import asyncio
import logging
import ptgctl
import ptgctl.util
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
#ptgctl.log.setLevel('WARNING')

configs = {'rule_classifier_path': '/src/app/models/recipe_tagger',
           'bert_classifier_path': '/src/app/models/bert_classifier'}

RECIPE_SID = 'event:recipe:id'
SESSION_SID = 'event:session:id'
UPDATE_STEP_SID = 'event:recipe:step'


class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

        self.state_manager = StateManager(configs)

    def start_recipe(self, recipe_id):
        logger.info('Starting recipe, ID=%s...' % str(recipe_id))
        if recipe_id is not None:
            recipe = self.api.recipes.get(recipe_id)
            logger.info('Loaded recipe: %s' % str(recipe))
            step_data = self.state_manager.start_recipe(recipe)
            logger.info('First step: %s' % str(step_data))

            return step_data

    async def run_reasoning(self, prefix='', top=5, use_clip=True):
        perception_actions_sid = f'{prefix}clip:action:steps' if use_clip else f'{prefix}egovlp:action:steps'
        perception_objects_sid = f'{prefix}detic:image'
        output_sid = f'{prefix}reasoning'

        async with self.api.data_pull_connect([perception_actions_sid, perception_objects_sid, RECIPE_SID, SESSION_SID, UPDATE_STEP_SID], ack=True) as ws_pull, \
                   self.api.data_push_connect([output_sid], batch=True) as ws_push:

            recipe_id = self.api.sessions.current_recipe()
            first_step = self.start_recipe(recipe_id)
            if first_step is not None:
                await ws_push.send_data([orjson.dumps(first_step)])

            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    if sid == RECIPE_SID:  # A call to start a new recipe
                        recipe_id = data.decode('utf-8')
                        first_step = self.start_recipe(recipe_id)
                        if first_step is not None:
                            await ws_push.send_data([orjson.dumps(first_step)])
                        continue
                    elif sid == SESSION_SID:  # A call to start a new session
                        self.state_manager.reset()
                        continue

                    elif sid == perception_objects_sid:  # A call sending objects and bounding boxes
                        #objects = orjson.loads(data)
                        #print('>>>>>>>>> objects', objects)
                        continue
                    elif sid == UPDATE_STEP_SID:  # A call to update the step
                        step_index = int(data)
                        updated_step = self.state_manager.set_user_feedback(step_index)
                        if updated_step is not None:
                            await ws_push.send_data([orjson.dumps(updated_step)])
                        continue

                    action_predictions = orjson.loads(data)
                    top_actions = sorted(action_predictions.items(), key=lambda x: x[1], reverse=True)[:top]
                    logger.info('Perception outputs: %s' % str(top_actions))
                    recipe_status = self.state_manager.check_status(top_actions)
                    logger.info('Reasoning outputs: %s' % str(recipe_status))
                    if recipe_status is not None:
                        await ws_push.send_data([orjson.dumps(recipe_status)])

    @ptgctl.util.async2sync
    async def run(self, *args, **kwargs):
        while True:
            try:
                await self.run_reasoning(*args, **kwargs)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3)


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)

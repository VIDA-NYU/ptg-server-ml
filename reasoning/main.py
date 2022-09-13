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

configs = {'rule_classifier_path': '/src/app/recipe_tagger',
           'bert_classifier_path': '/src/app/bert_classifier'}

RECIPE_SID = 'event:recipe:id'
SESSION_SID = 'event:session:id'


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

    async def run_reasoning(self, prefix='', top=5):
        input_sid = f'{prefix}clip:action:steps'
        output_sid = f'{prefix}reasoning'

        async with self.api.data_pull_connect([input_sid, RECIPE_SID, SESSION_SID], rate_limit=1) as ws_pull, \
                   self.api.data_push_connect([output_sid], batch=True) as ws_push:

            recipe_id = self.api.sessions.current_recipe()
            self.start_recipe(recipe_id)

            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    if sid == RECIPE_SID:
                        recipe_id = data.decode('utf-8')
                        self.start_recipe(recipe_id)
                        continue
                    if sid == SESSION_SID:
                        self.state_manager.reset()
                        logger.info('Recipe resetted')
                        continue

                    action_predictions = orjson.loads(data)
                    top_actions = sorted(action_predictions.items(), key=lambda x: x[1], reverse=True)[:top]
                    logger.info('Perception outputs: %s' % str(top_actions))
                    recipe_status = self.state_manager.check_status([i[0] for i in top_actions])
                    logger.info('Reasoning outputs: %s' % str(recipe_status))

                    await ws_push.send_data([orjson.dumps(recipe_status)])

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        while True:
            try:
                await self.run_reasoning(*a, **kw)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3)


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)

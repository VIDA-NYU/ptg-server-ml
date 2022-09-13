import os
import time
import orjson
import asyncio
import logging
import ptgctl
import ptgctl.holoframe
import ptgctl.util
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

#ptgctl.log.setLevel('WARNING')

configs = {'rule_classifier_path': '/src/app/recipe_tagger',
           'bert_classifier_path': '/src/app/bert_classifier'}


class ReasoningApp:
    RECIPE_SID = 'event:recipe:id'
    RECIPE_STEP_SID = 'event:recipe:step'
    SESSION_SID = 'event:session:id'

    def __init__(self):
        self.api = ptgctl.CLI(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

        self.state_manager = StateManager(configs)

    def change_recipe(self, recipe_id):
        if not recipe_id:
            self.state_manager.reset()
            return None, None

        recipe = self.api.recipes.get(recipe_id)
        logger.info('Loaded recipe: %s' % str(recipe))
        step_data = self.state_manager.start_recipe(recipe)
        logger.info('First step: %s' % str(step_data))
        return step_data, recipe

    async def _run(self, prefix=None, top=5):
        prefix = prefix or ''
        input_sid = f'{prefix}clip:action:steps'

        async with self.api.data_pull_connect([input_sid, self.RECIPE_SID, self.RECIPE_STEP_SID, self.SESSION_SID], rate_limit=1) as ws_pull, \
                   self.api.data_push_connect([f'{prefix}reasoning'], batch=True) as ws_push:

            recipe_id: str = self.api.sessions.current_recipe()
            step_data, recipe = self.change_recipe(recipe_id)

            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    if sid == self.RECIPE_SID:
                        self.change_recipe(data.decode('utf-8'))
                        continue
                    if sid == self.RECIPE_STEP_SID:
                        #self.override_recipe_step(int(data))
                        continue
                    if sid == self.SESSION_SID:
                        self.state_manager.reset()
                        continue

                    action_predictions = orjson.loads(data)
                    top_actions = sorted(action_predictions.items(), key=lambda x: x[1], reverse=True)[:top]
                    logger.info('Perception outputs: %s' % str(top_actions))
                    recipe_status = self.state_manager.check_status([i[0] for i in top_actions])
                    logger.info('Reasoning outputs: %s' % str(recipe_status))

                    #if False: ## step changed
                    #    self.api.sessions.update_step(X)

                    await ws_push.send_data([orjson.dumps(recipe_status)])

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        while True:
            try:
                await self._run(*a, **kw)
            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3)


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)

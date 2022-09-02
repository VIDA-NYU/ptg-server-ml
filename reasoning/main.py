import os
import orjson
import logging
import ptgctl
import ptgctl.holoframe
import ptgctl.util
import numpy as np
from tim_reasoning import StateManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

#ptgctl.log.setLevel('WARNING')

configs = {'rule_classifier_path': '/src/app/recipe_tagger',
           'bert_classifier_path': '/src/app/bert_classifier'}


class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.CLI(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

    @ptgctl.util.async2sync
    async def run(self, prefix=None):
        prefix = prefix or ''
        input_sid = f'{prefix}clip:action:steps'

        recipe_id: str = self.api.sessions.current_recipe()
        recipe = self.api.recipes.get(recipe_id)
        logger.info('Loaded recipe: %s' % str(recipe))
        state_manager = StateManager(configs)
        step_data = state_manager.start_recipe(recipe)
        logger.info('First step: %s' % str(step_data))

        async with self.api.data_pull_connect(input_sid) as ws_pull, \
                   self.api.data_push_connect([f'{prefix}reasoning'], batch=True) as ws_push:
            while True:
                data = ptgctl.holoframe.load_all(await ws_pull.recv_data())
                action_pred = data[input_sid]
                action_pred.pop('timestamp', None)
                action_pred.pop('frame_type', None)
                action_text, action_prob = zip(*action_pred.items())
                logger.info('Perception outputs: %s' % str(action_text))
                i_topk = np.argpartition(action_prob, 5)
                i_topk = sorted(i_topk, key=lambda i: action_prob[i], reverse=True)
                results = state_manager.check_status([action_text[i] for i in i_topk])
                logger.info('Reasoning outputs: %s' % str(results))

                await ws_push.send_data([orjson.dumps(results)])


if __name__ == '__main__':
    import fire
    fire.Fire(ReasoningApp)

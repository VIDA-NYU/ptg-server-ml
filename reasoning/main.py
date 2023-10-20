import os
import nltk
import orjson
import asyncio
import logging
import ptgctl
import ptgctl.util
from tim_reasoning import SessionManager


MODEL_DIR = os.getenv('MODEL_DIR') or 'models'
NLTK_DIR = os.path.join(MODEL_DIR, 'nltk')
nltk.data.path.append(NLTK_DIR)
nltk.download('punkt', download_dir=NLTK_DIR)

RECIPE_SID = 'event:recipe:id'
SESSION_SID = 'event:session:id'
OBJECT_STATES_SID = 'detic:image'
UPDATE_STEP_SID = 'event:recipe:step'
REASONING_STATUS_SID = 'reasoning:check_status'

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

        self.session_manager = SessionManager(patience=1)

    def start_recipe(self, recipe_id):
        pass

    async def run_reasoning(self, prefix=''):
        object_states_sid = prefix + OBJECT_STATES_SID
        re_check_status_sid = prefix + REASONING_STATUS_SID

        async with self.api.data_pull_connect([object_states_sid, RECIPE_SID, SESSION_SID, UPDATE_STEP_SID], ack=True) as ws_pull, \
                   self.api.data_push_connect([re_check_status_sid], batch=True) as ws_push:

            detected_object_states = None

            while True:
                for sid, timestamp, data in await ws_pull.recv_data():

                    if sid == RECIPE_SID:  # A call to start a new recipe
                        print('New recipe')
                        continue

                    elif sid == UPDATE_STEP_SID:  # A call to update the step
                        print('Updating step', data)
                        continue

                    elif sid == SESSION_SID:  # A call to start a new session
                        #self.state_manager.reset()
                        continue

                    elif sid == object_states_sid:  # A call sending detected object states
                        detected_object_states = orjson.loads(data)
                        logger.info(f'Perception outputs: {str(detected_object_states)}')

                    if detected_object_states is not None and len(detected_object_states) > 0:
                        for entry in detected_object_states:
                            entry['id'] = entry['segment_track_id']
                            recipe_status = self.session_manager.handle_message(message=[entry])[0]

                            if recipe_status is not None:
                                logger.info(f'Reasoning outputs: {str(recipe_status)}')
                                recipe_status['step_id'] = int(recipe_status['step_id'])
                                await ws_push.send_data([orjson.dumps(recipe_status)], re_check_status_sid)
                                # Reset the values of the detected inputs
                                detected_object_states = None


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


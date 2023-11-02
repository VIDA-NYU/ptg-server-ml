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

OBJECT_STATES_SID = 'detic:memory'
UPDATE_STEP_SID = 'arui:change_step'
UPDATE_TASK_SID = 'arui:change_task'
PAUSE_SID = 'arui:pause'
RESET_SID = 'arui:reset'
REASONING_STATUS_SID = 'reasoning:check_status'

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class ReasoningApp:

    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'reasoning',
                              password=os.getenv('API_PASS') or 'reasoning')

        self.session_manager = SessionManager(patience=1)
        self.pause = False

    async def run_reasoning(self, prefix=''):
        object_states_sid = prefix + OBJECT_STATES_SID
        re_check_status_sid = prefix + REASONING_STATUS_SID

        async with self.api.data_pull_connect([object_states_sid, UPDATE_TASK_SID, UPDATE_STEP_SID, PAUSE_SID,
                                               RESET_SID], ack=True) as ws_pull, \
                self.api.data_push_connect([re_check_status_sid], batch=True) as ws_push:

            detected_object_states = None

            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    if sid == UPDATE_STEP_SID:  # A call to update the step
                        data = data.decode('utf-8')
                        logger.info(f'Update step: {data}')
                        step_id, task_id = data.split('&')
                        updated_step = self.session_manager.update_step(int(task_id), int(step_id))
                        logger.info(f'Updated step: {str(updated_step)}')

                        if updated_step is not None:
                            await ws_push.send_data([orjson.dumps(updated_step)], re_check_status_sid)
                        continue

                    elif sid == UPDATE_TASK_SID:  # A call to update the task
                        data = data.decode('utf-8')
                        logger.info(f'Update task: {data}')
                        task_name, task_id = data.split('&')
                        updated_task = self.session_manager.update_task(int(task_id), task_name)
                        logger.info(f'Updated task: {str(updated_task)}')

                        if updated_task is not None:
                            await ws_push.send_data([orjson.dumps(updated_task)], re_check_status_sid)
                        continue

                    elif sid == RESET_SID:  # A call to reset the session
                        logger.info(f'Reset session')
                        self.session_manager = SessionManager(patience=1)
                        self.pause = False

                    elif sid == RESET_SID:  # A call to pause/resume the session
                        status = data.decode('utf-8')
                        logger.info(f'Pause/resume session: {status}')
                        if status == 'pause':
                            self.pause = True
                        else:
                            self.pause = False

                    elif sid == object_states_sid:  # A call sending detected object states
                        detected_object_states = orjson.loads(data)
                        logger.info(f'Perception outputs: {str(detected_object_states)}')

                    if not self.pause and detected_object_states is not None and len(detected_object_states) > 0:
                        for entry in detected_object_states:
                            task_status = self.session_manager.handle_message(message=[entry])
                            logger.info(f'Reasoning outputs: {str(task_status)}')
                            if task_status['active_tasks'][0] is not None:
                                await ws_push.send_data([orjson.dumps(task_status)], re_check_status_sid)
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

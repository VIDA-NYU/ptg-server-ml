import os
import orjson
import asyncio
from collections import OrderedDict, defaultdict

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np

import ptgctl
from ptgctl import holoframe
import ptgctl.util
from ptgprocess.detic import Detic

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

#ptgctl.log.setLevel('WARNING')


DEFAULT_VOCAB = 'lvis'

class DeticApp:
    image_box_keys = ['xyxyn', 'confidence', 'class_id', 'label']
    vocab_keys = ['tools_simple', 'ingredients_simple', 'objects']

    EXTRA_VOCAB = ['person', 'feet']

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'detic',
                              password=os.getenv('API_PASS') or 'detic')
        self.model = Detic(device='cuda:0', **kw)

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        while True:
            try:
                await self._run(*a, **kw)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)
    
    async def _run(self):
        # stream ids
        in_sids = ['main']
        out_sids = ['detic:image', 'detic:image:for3d', 'detic:hands']
        recipe_sid = 'event:recipe:id'

        mean_person_confs_decay = 0
        decay = 0.5

        self.change_recipe(self.api.sessions.current_recipe())

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect(in_sids + [recipe_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, buffer in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        tms = int(t.split('-')[0])
                        
                        # watch for recipe changes
                        if sid == recipe_sid:
                            await writer.write(b'[]', b'[]')
                            self.change_recipe(buffer.decode('utf-8'))
                            continue
                        
                        # compute box
                        main = holoframe.load(buffer)
                        im = main['image']
                        xyxy, confs, class_ids, labels = self.predict(im)
                        xyxyn = boxnorm(xyxy, *im.shape[:2])

                        # use person detection as a reasoning signal
                        is_person = labels == 'person'
                        person_confs = confs[is_person] if is_person.sum() else np.zeros((1,))
                        mean_person_confs = np.nanmean(person_confs)
                        mean_person_confs_decay = (1 - decay) * mean_person_confs + decay * mean_person_confs_decay

                        objs = self.zip_objs(
                            self.image_box_keys,
                            [xyxyn, confs, class_ids, labels])
                        await ws_push.send_data([
                            jsondump(objs),
                            jsondump({'objects': objs, 'image_params': {
                                'shape': main['image'].shape,
                                'focal': [main['focalX'], main['focalY']], 
                                'principal': [main['principalX'], main['principalY']],
                                'cam2world': main['cam2world'].tolist(),
                            }}),
                            jsondump({
                                'n_person_boxes': is_person.sum(),
                                'max_conf': np.max(person_confs, initial=0),
                                'mean_conf': mean_person_confs,
                                'mean_conf_smooth': mean_person_confs_decay,
                            }),
                        ], out_sids, [t, t, t])
                        
    def change_recipe(self, recipe_id):
        if not recipe_id:
            print('no recipe. using default vocab:', DEFAULT_VOCAB)
            self.model.set_vocab(DEFAULT_VOCAB)
            return
        print('using recipe', recipe_id)
        recipe = self.api.recipes.get(recipe_id)
        self.model.set_vocab([w for k in self.vocab_keys for w in recipe.get(k) or []] + self.EXTRA_VOCAB)

    def predict(self, im):
        outputs = self.model(im)
        insts = outputs["instances"].to("cpu")
        xyxy = insts.pred_boxes.tensor.numpy()
        class_ids = insts.pred_classes.numpy().astype(int)
        confs = insts.scores.numpy()
        labels = self.model.labels[class_ids]
        return xyxy, confs, class_ids, labels

    def zip_objs(self, keys, xs):
        return [dict(zip(keys, xs)) for xs in zip(*xs)]


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)



def boxnorm(xyxy, h, w):
    xyxy[:, 0] = (xyxy[:, 0]) / w
    xyxy[:, 1] = (xyxy[:, 1]) / h
    xyxy[:, 2] = (xyxy[:, 2]) / w
    xyxy[:, 3] = (xyxy[:, 3]) / h
    return xyxy


if __name__ == '__main__':
    import fire
    fire.Fire(DeticApp)

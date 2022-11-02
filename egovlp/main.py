import os
import orjson
import asyncio
from collections import OrderedDict, defaultdict, deque

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np

import ptgctl
from ptgctl import holoframe
import ptgctl.util
from ptgprocess.egovlp import EgoVLP

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

#ptgctl.log.setLevel('WARNING')

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


import os
import glob
import numpy as np
import torch
from torch import nn


RECIPE_EXAMPLES_DIR = '/src/app/fewshot'


class ZeroShotPredictor(nn.Module):
    def __init__(self, vocab, model, prompt='{}'):
        super().__init__()
        self.model = model
        self.vocab = np.asarray(vocab)
        tqdm.tqdm.write(f'zeroshot with: {vocab}')
        self.Z_vocab = self.model.encode_text(vocab, prompt)

    def forward(self, Z_image):
        '''Returns the action probabilities for that frame.'''
        scores = self.model.similarity(self.Z_vocab, Z_image).detach()
        return scores

class FewShotPredictor(nn.Module):
    def __init__(self,  vocab_dir, n_neighbors=33):
        super().__init__()

        # load all the data
        assert os.path.isdir(vocab_dir)
        fsx = sorted(glob.glob(os.path.join(vocab_dir, 'X_*.npy')))
        fsy = sorted(glob.glob(os.path.join(vocab_dir, 'Y_*.npy')))
        assert all(
            os.path.basename(fx).split('_', 1)[1] == os.path.basename(fy).split('_', 1)[1]
            for fx, fy in zip(fsx, fsy)
        )
        fvocab = os.path.join(vocab_dir, 'classes.npy')

        # load and convert to one big numpy array
        X = np.concatenate([np.load(f) for f in fsx])
        Y = np.concatenate([np.load(f) for f in fsy])
        self.vocab = vocab = np.asarray(np.load(fvocab))
        tqdm.tqdm.write(f'loaded {X.shape} {Y.shape}. {len(vocab)} {vocab}')

        # train the classifier
        tqdm.tqdm.write('training classifier...')
        #self.clsf = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.clsf = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
        self.clsf.fit(X, Y)
        print(self.clsf.classes_)
        tqdm.tqdm.write('trained!')

    def forward(self, Z_image):
        scores = self.clsf.predict_proba(Z_image.cpu().numpy())
        return scores

def l2norm(Z, eps=1e-8):
    return Z / np.maximum(np.linalg.norm(Z, keepdims=True), eps)

def normalize(Z, eps=1e-8):
    Zn = Z.norm(dim=1)[:, None]
    return Z / torch.max(Zn, eps*torch.ones_like(Zn))






class EgoVLPApp:
    vocab_key = 'steps_simple'
    decay = 0.5

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'egovlp',
                              password=os.getenv('API_PASS') or 'egovlp')
        self.model = EgoVLP(**kw)
        self.q = deque(maxlen=16)

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        while True:
            try:
                recipe_id = self.api.session.current_recipe()
                while not recipe_id:
                    print("waiting for recipe to be activated")
                    recipe_id = await self._watch_recipe_id(recipe_id)
                
                print("Starting recipe:", recipe_id)
                await self._run(recipe_id, *a, **kw)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_recipe_id(self, recipe_id):
        async with self.api.data_pull_connect(self.RECIPE_SID) as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != recipe_id:
                        return data

    async def _run(self, recipe_id):
        self.predictor = None

        # call the api and get the recipe vocab text
        self.set_recipe(recipe_id)

        # stream ids
        in_sid = 'main'
        recipe_sid = 'event:recipe:id'
        vocab_sid = 'event:recipes'

        out_sid = 'egovlp:action:steps'

        sim_decay = 0

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect([out_sid], batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        tms = int(t.split('-')[0])

                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            if not d: break
                            self.set_recipe(d.decode('utf-8'))
                            sim_decay = 0
                            continue
                        if not self.predictor:
                            print("no actions selected", flush=True)
                            return

                        # encode the image
                        d = holoframe.load(d)
                        im = d['image']
                        self.q.append(self.model.prepare_image(im))
                        vid = torch.stack(list(self.q), dim=1).to(self.model.device)
                        z_image = self.model.encode_video(vid)

                        sim = self.predictor(z_image)[0]
                        #sim_decay = sim = (1 - self.decay) * sim + self.decay * sim_decay
                    
                        # send data
                        await ws_push.send_data(
                            [jsondump(dict(zip(self.predictor.vocab.tolist(), sim.tolist())))], 
                            [out_sid], [t])

    def set_recipe(self, recipe_id):
        if not recipe_id:
            print('no recipe.', None)
            self.predictor = None
            return
        print('using recipe', recipe_id)
        
        recipe_dir = os.path.join(RECIPE_EXAMPLES_DIR, recipe_id)
        if os.path.isdir(recipe_dir):
            tqdm.tqdm.write('few shot')
            self.predictor = FewShotPredictor(recipe_dir)
        else:
            tqdm.tqdm.write('zero shot')
            recipe = self.api.recipes.get(recipe_id) or {}
            steps = recipe.get(self.vocab_key)
            if not steps:
                print(f"\nRecipe {recipe_id}.{self.vocab_key} returned no steps.")
                self.predictor = None
                return

            self.predictor = ZeroShotPredictor(steps, self.model)


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(EgoVLPApp)

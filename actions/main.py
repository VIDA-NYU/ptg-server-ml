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

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

#ptgctl.log.setLevel('WARNING')

# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# import pickle

# import glob
import torch
# from torch import nn


RECIPE_EXAMPLES_DIR = '/src/app/fewshot'
RECORDING_RAW_DIR = '/src/app/recordings/raw'
RECORDING_POST_DIR = '/src/app/recordings/post'


# class ZeroShotPredictor(nn.Module):
#     def __init__(self, vocab, model, prompt='{}'):
#         super().__init__()
#         self.model = model
#         self.vocab = np.asarray(vocab)
#         tqdm.tqdm.write(f'zeroshot with: {vocab}')
#         self.Z_vocab = self.model.encode_text(vocab, prompt)

#     def forward(self, Z_image):
#         '''Returns the action probabilities for that frame.'''
#         scores = self.model.similarity(self.Z_vocab, Z_image).detach()
#         return scores

# class FewShotPredictor(nn.Module):
#     def __init__(self,  vocab_dir, **kw):
#         super().__init__()
#         self._load_model(vocab_dir, **kw)

#     def _load_model(self, vocab_dir, clsf_type='knn', n_neighbors=33):
#         pkl_fname = f'{vocab_dir}_{clsf_type}.pkl'
#         if os.path.isfile(pkl_fname):
#             with open(pkl_fname, 'rb') as fh:
#                 tqdm.tqdm.write('loading classifier...')
#                 self.clsf, self.vocab = pickle.load(fh)
#                 tqdm.tqdm.write(f'loaded classifier from disk. {len(self.vocab)} {self.vocab}')
#             return

#         # load all the data
#         assert os.path.isdir(vocab_dir)
#         fsx = sorted(glob.glob(os.path.join(vocab_dir, 'X_*.npy')))
#         fsy = sorted(glob.glob(os.path.join(vocab_dir, 'Y_*.npy')))
#         assert all(
#             os.path.basename(fx).split('_', 1)[1] == os.path.basename(fy).split('_', 1)[1]
#             for fx, fy in zip(fsx, fsy)
#         )
#         fvocab = os.path.join(vocab_dir, 'classes.npy')

#         # load and convert to one big numpy array
#         X = np.concatenate([np.load(f) for f in fsx])
#         Y = np.concatenate([np.load(f) for f in fsy])
#         self.vocab = vocab = np.asarray(np.load(fvocab))
#         tqdm.tqdm.write(f'loaded {X.shape} {Y.shape}. {len(vocab)} {vocab}')

#         # train the classifier
#         tqdm.tqdm.write('training classifier...')
#         if clsf_type == 'knn':
#             self.clsf = KNeighborsClassifier(n_neighbors=n_neighbors)
#         elif clsf_type == 'xgb':
#             self.clsf = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
#         else:
#             raise ValueError(f"Invalid classifier {clsf_type}")
#         self.clsf.fit(X, Y)
#         tqdm.tqdm.write('trained!')

#         with open(pkl_fname, 'wb') as fh:
#             pickle.dump([self.clsf, self.vocab], fh)

#     def forward(self, Z_image):
#         scores = self.clsf.predict_proba(Z_image.cpu().numpy())
#         return scores

# def l2norm(Z, eps=1e-8):
#     return Z / np.maximum(np.linalg.norm(Z, keepdims=True), eps)

# def normalize(Z, eps=1e-8):
#     Zn = Z.norm(dim=1)[:, None]
#     return Z / torch.max(Zn, eps*torch.ones_like(Zn))


# class ShotEgoVLP(EgoVLP):
#     predictor = None
#     # def __init__(self, *a, **kw):
#     #     super().__init__(*a, **kw)
    
#     def set_vocab(self, vocab, few_shot_dir, step_map=None):
#         self.predictor = self._get_predictor(vocab, few_shot_dir, step_map)
#         assert self.predictor is not None
#         self.vocab = self.predictor.vocab

#     def _get_predictor(self, vocab, few_shot_dir, step_map=None):
#         if os.path.isdir(few_shot_dir):
#             tqdm.tqdm.write('few shot')
#             self.predictor = FewShotPredictor(few_shot_dir)

#         tqdm.tqdm.write('zero shot')
#         if not vocab:
#             raise ValueError(f'no vocab for recipe {os.path.basename(few_shot_dir)}')
#         self.predictor = ZeroShotPredictor(vocab, self)
#         if step_map:
#             self.predictor.vocab = np.array([step_map.get(x,x) for x in self.predictor.vocab])

#     def predict_video(self, vid):
#         assert self.predictor is not None
#         z_image = self.model.encode_video(vid)
#         sim = self.predictor(z_image)
#         return sim


class ActionSession:
    vocab_key = 'steps_simple'
    decay = 0.5
    def __init__(self, model, recipe_id, vocab, step_map):
        self.model = model
        model.set_vocab(vocab, os.path.join(RECIPE_EXAMPLES_DIR, recipe_id), step_map=step_map)
        self.q = deque(maxlen=16)
        self.sim_decay = 0

    def on_image(self, image, **extra):
        # add the image to the frame queue
        self.q.append(self.model.prepare_image(image))
        vid = torch.stack(list(self.q), dim=1).to(self.model.device)
        sim = self.model.predict_video(vid)[0]
        #sim_decay = sim = (1 - self.decay) * sim + self.decay * sim_decay

        # return as a class->probability dictionary
        return dict(zip(
            self.model.vocab.tolist(), 
            sim.tolist(),
        ))




class RecipeExit(Exception):
    pass





class ActionApp:
    vocab_key = 'steps_simple'
    decay = 0.5

    def __init__(self, model_name='egovlp', **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'egovlp',
                              password=os.getenv('API_PASS') or 'egovlp')
        if model_name == 'egovlp':
            from ptgprocess.egovlp import ShotEgoVLP
            self.model = ShotEgoVLP(**kw)
        elif model_name == 'omnivore':
            from ptgprocess.omnivore import Omnivore
            self.model = Omnivore(**kw)
        self.q = deque(maxlen=16)

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
            except RecipeExit:
                pass
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_recipe_id(self, recipe_id):
        async with self.api.data_pull_connect('event:recipe:id') as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != recipe_id:
                        return data

    def start_session(self, recipe_id):
        '''Initialize the action session for a recipe.'''
        if not recipe_id:
            raise RecipeExit("no recipe set.")
        recipe = self.api.recipes.get(recipe_id) or {}
        vocab = recipe.get(self.vocab_key)
        step_map = recipe.get('step_map') or {}
        if not vocab:
            raise RecipeExit(f"\nRecipe {recipe_id}.{self.vocab_key} returned no steps.")
        self.session = ActionSession(self.model, recipe_id, vocab, step_map)

    async def run_recipe(self, recipe_id, prefix=None):
        '''Run the recipe.'''
        self.start_session(recipe_id)

        # stream ids
        in_sid = f'{prefix or ""}main'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'
        out_sid = f'{prefix or ""}egovlp:action:steps'

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect([out_sid], batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            return 
                            # self.start_session(d.decode('utf-8'))
                            # continue

                        # predict actions
                        preds = self.session.on_image(**holoframe.load(d))
                        await ws_push.send_data([jsondump(preds)], [out_sid], [t])

    def run_offline(self, recording_id, recipe_id, out_sid='egovlp:action:steps'):
        from ptgprocess.record import RawReader, RawWriter, JsonWriter
        self.start_session(recipe_id)

        raw_dir = os.path.join(RECORDING_RAW_DIR, recording_id)
        post_dir = os.path.join(RECORDING_POST_DIR, recording_id)
        print("raw directory:", raw_dir)
        with RawReader(os.path.join(raw_dir, 'main')) as reader, \
             RawWriter(out_sid, raw_dir) as writer, \
             JsonWriter(out_sid, post_dir) as json_writer:
 
            for ts, d in reader:
                sim_dict = self.session.on_image(**holoframe.load(d))
                writer.write(jsondump(sim_dict), ts)
                json_writer.write(jsondump({**sim_dict, 'timestamp': ts}), ts)




def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(ActionApp)

import os
import orjson
import asyncio
import functools
from collections import OrderedDict, defaultdict, deque

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
from torch import nn
from torch.nn import functional as F

import ptgctl
from ptgctl import holoframe
import ptgctl.util

import librosa

from ptgprocess.omnivore import Omnivore
from ptgprocess.audio_slowfast import AudioSlowFast
from ptgprocess.omnimix import OmniGRU

from steps import get_step_labels

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

import torch
# from torch import nn



RECORDING_RAW_DIR = '/src/app/recordings/raw'
RECORDING_POST_DIR = '/src/app/recordings/post'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StepsSession:
    vocab_key = 'steps_simple'
    decay = 0.5
    audio_sr = 24000
    emb_stride = 2
    emb_len = 8
    def __init__(self, model, vocab, id_vocab, step_mask, temperature=0.3, prefix=None):
        self.model = model
        self.rgb_q = deque(maxlen=32)
        self.aud_q = deque(maxlen=16)
        self.rgb_emb_q = deque(maxlen=8 * self.emb_stride)
        self.aud_emb_q = deque(maxlen=8 * self.emb_stride)
        self.step_vocab = vocab
        self.step_id_vocab = id_vocab
        self.step_mask = torch.Tensor(step_mask).long()
        self.sim_decay = 0
        prefix = f'{prefix or ""}omnimix'
        self.temperature = temperature

        self.step_sid = f'{prefix}:step'
        self.step_id_sid = f'{prefix}:step:id'
        self.rgb_verb_sid = f'{prefix}:verb:rgb'
        self.audio_verb_sid = f'{prefix}:verb:audio'
        self.out_sids = [
            self.step_sid,
            self.step_id_sid,
            # f'{prefix}:action:rgb',
            self.rgb_verb_sid,
            # f'{prefix}:noun:rgb',
            self.audio_verb_sid,
            # f'{prefix}:noun:audio',
        ]

        # create initial embeddings
        self.hidden = None

        # empty rgb embedding
        x = torch.zeros(1, 3, 32, 224, 224).to(device)
        _, emb_vid = self.model.rgb(x, return_embedding=True)
        self.empty_rgb_emb = emb_vid#torch.zeros(1, 1024).to(device)

        # empty audio embedding
        if self.model.mix.use_audio:
            y = self.model.aud.prepare_audio(np.random.randn(int(1.999 * self.audio_sr)) * 1e-4, self.audio_sr)
            y = [y.to(device) for y in y]
            _, emb_aud = self.model.aud(y, return_embedding=True)
            self.empty_aud_emb = emb_aud

    def on_image(self, image, **extra):
        # add the image to the frame queue
        self.rgb_q.append(self.model.rgb.prepare_image(image))
        verb, noun = self._predict_video()
        steps = self._predict_step()
        # print(22222, verb.shape, noun.shape, steps.shape)
        # print(steps, self.step_vocab[torch.argmax(steps)], flush=True)
        return {
            self.step_sid: dict(zip(self.step_vocab.tolist(), steps.detach().cpu().tolist())) if steps is not None else None, 
            self.step_id_sid: dict(zip(map(str, self.step_id_vocab.tolist()), steps.detach().cpu().tolist())) if steps is not None else None, 
            self.rgb_verb_sid: dict(zip(self.model.aud.vocab[0], verb[0].detach().cpu().tolist())),
        }

    def on_audio(self, audio, sr, pos, **extra):
        # return
        # add the image to the frame queue
        # self.audio_sr = sr
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sr)
        self.aud_q.append(audio[:, 0])
        verb, noun = self._predict_audio()
        # steps = self._predict_step()
        # print('audio', steps.shape if steps is not None else None, verb.shape, noun.shape)
        return {
            # self.step_sid: dict(zip(self.step_vocab.tolist(), steps.detach().cpu().tolist())) if steps is not None else None, 
            # self.step_id_sid: dict(zip(self.step_id_vocab.tolist(), steps.detach().cpu().tolist())) if steps is not None else None, 
            self.audio_verb_sid: dict(zip(self.model.aud.vocab[0], verb[0].detach().cpu().tolist())) if verb is not None else None,
        }

    def _predict_video(self):
        vid = torch.stack(list(self.rgb_q) or [torch.zeros(3, 224, 224)], dim=1).to(device)[None]
        verb_rgb, noun_rgb, emb_rgb = self.model.video(vid)
        self.rgb_emb_q.append(emb_rgb)
        return verb_rgb, noun_rgb
    
    def _predict_audio(self):
        if not self.aud_q:
            return None, None
        # aud = [
        #     torch.concat(list(path), dim=0).to(device)[None]
        #     for path in list(zip(*(self.aud_q)))
        # ]
        aud = np.concatenate(list(self.aud_q))
        if len(aud) < self.audio_sr * 1.999:
            return None, None
        # print(aud.shape)
        aud = aud[-int(1.999*self.audio_sr):]
        # print(aud.shape, aud.min(), aud.max())
        spec = self.model.aud.prepare_audio(aud, self.audio_sr)
        spec = [x.to(device) for x in spec]
        (verb_aud, noun_aud), emb_aud = self.model.aud(spec, return_embedding=True)
        # print(emb_aud.shape)
        self.aud_emb_q.append(emb_aud)
        return verb_aud, noun_aud
    
    def _predict_step(self):
        emb_rgbs = self._prepare_window(self.rgb_emb_q, self.empty_rgb_emb)
        if self.model.mix.use_audio:
            emb_auds = self._prepare_window(self.aud_emb_q, self.empty_aud_emb)
            steps, self.hidden = self.model.mix(rgb=emb_rgbs, aud=emb_auds, h=self.hidden)
        else:
            steps, self.hidden = self.model.mix(rgb=emb_rgbs, h=self.hidden)
        # print(1, steps.shape)
        steps = F.softmax(steps[0, self.step_mask] * self.temperature, dim=-1)
        # print(2, steps.shape)
        return steps

    def _prepare_window(self, q, empty_emb):
        Z = list(q)[::self.emb_stride]
        Z = torch.stack([empty_emb]*(self.emb_len - len(Z)) + list(Z), dim=1)
        return Z.to(device)


class Omnimix(nn.Module):
    def __init__(self, skill):
        super().__init__()
        self.mix = get_omnimix(skill) #OmniGRU(skill)
        self.rgb = get_omnivore() #Omnivore()
        # if self.mix.use_audio:
        self.aud = get_asf() #AudioSlowFast()

    def video(self, im):
        action_rgb, emb_rgb = self.rgb(im, return_embedding=True)
        verb_rgb, noun_rgb = self.rgb.project_verb_noun(action_rgb)
        return verb_rgb, noun_rgb, emb_rgb
    
    # def audio(self, aud):
    #     (verb_aud, noun_aud), emb_aud = self.aud(aud, return_embedding=True)
    #     return verb_aud, noun_aud, emb_aud

    # def forward(self, im, aud):
    #     action_rgb, emb_rgb = self.rgb(im, return_embedding=True)
    #     verb_rgb, noun_rgb = self.rgb.project_verb_noun(action_rgb)

    #     (verb_aud, noun_aud), emb_aud = self.aud(aud)
    #     step = self.mix(emb_rgb, emb_aud)
    #     return step, action_rgb, (verb_rgb, noun_rgb), (verb_aud, noun_aud)


@functools.lru_cache(1)
def get_omnivore():
    return Omnivore()

@functools.lru_cache(1)
def get_asf():
    return AudioSlowFast()

@functools.lru_cache(1)
def get_omnimix(skill):
    return OmniGRU(skill)



class RecipeExit(Exception):
    pass



class StepsApp:
    vocab_key = 'steps_simple'
    decay = 0.5

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'omnimix',
                              password=os.getenv('API_PASS') or 'omnimix')

    @torch.no_grad()
    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        '''Persistent running app, with error handling and recipe status watching.'''
        while True:
            try:
                skill_id = self.api.session.current_recipe()
                while not skill_id:# or skill_id == 'tourniquet'
                    print("waiting for recipe to be activated")
                    skill_id = await self._watch_skill_id(skill_id)
                
                print("Starting recipe:", skill_id)
                await self.run_recipe(skill_id, *a, **kw)
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_skill_id(self, skill_id=b''):
        skill_id = skill_id or b''
        async with self.api.data_pull_connect('event:recipe:id') as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != skill_id:
                        return data.decode('utf-8')

    def start_session(self, skill_id, prefix=None):
        '''Initialize the action session for a recipe.'''
        if not skill_id:
            raise RecipeExit("no recipe set.")
        # recipe = self.api.recipes.get(skill_id) or {}
        # vocab = recipe.get(self.vocab_key)

        # step_map = recipe.get('step_map') or {}
        # if not vocab:
        #     raise Exception(f"\nRecipe {skill_id}.{self.vocab_key} returned no steps in {set(recipe)}.")


        self.model = Omnimix(skill_id).cuda()
        self.model.eval()  # LOL fuck me. forgetting this is a bad time
        
        # TODO: this should come from the model config/mongodb
        print(self.model.mix.skills)
        step_vocab, step_ids, step_masks = get_step_labels(self.model.mix.skills)
        if skill_id not in step_masks:
            raise RecipeExit(f"{skill_id} not supported by this model.")
        
        print(skill_id, flush=True)
        print(step_masks, flush=True)
        print(step_vocab, flush=True)
        print(step_ids, flush=True)
        step_mask = step_masks[skill_id]
        print(step_mask, flush=True)
        step_vocab = step_vocab[step_mask]
        step_ids = step_ids[step_mask]
        self.session = StepsSession(self.model, step_vocab, step_ids, step_mask, prefix=prefix)

    async def run_recipe(self, skill_id, prefix=None):
        '''Run the recipe.'''
        self.start_session(skill_id, prefix=prefix)

        # stream ids
        # in_rgb_sid = 'main'
        # in_aud_sid = 'mic0'
        in_rgb_sid = f'{prefix or ""}main'
        in_aud_sid = f'{prefix or ""}mic0'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        print(self.session.out_sids)

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([in_rgb_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(self.session.out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            print("recipe changed", skill_id, '->', d, flush=True)
                            # self.start_session(skill_id, prefix=prefix)
                            return

                        # predict actions
                        preds = None
                        if sid == in_rgb_sid:
                            preds = self.session.on_image(**holoframe.load(d))
                        #elif sid == in_aud_sid:
                        #    preds = self.session.on_audio(**holoframe.load(d))
                        if preds is not None:
                            preds = {k: v for k, v in preds.items() if v is not None}
                            await ws_push.send_data(
                                [jsondump({k: round(v, 3) for k, v in sorted(x.items(), key=lambda x: -x[1])}) for x in preds.values()], 
                                list(preds.keys()), 
                                #[noconfliict_ts(t)]*len(preds)
                            )

    @torch.no_grad()
    def run_offline(self, recording_id, skill_id, prefix=None):
        from contextlib import ExitStack
        from ptgprocess.util import VideoInput
        from ptgprocess.record import RawReader, RawWriter, JsonWriter
        self.start_session(skill_id, prefix=prefix)

        with ExitStack() as exits:
            raw_writer = None
            if os.path.isfile(recording_id):
                loader = lambda x: {'image': x, 'focalX': 0, 'focalY': 0, 'principalX': 0, 'principalY': 0, 'cam2world': np.zeros((4,4))}
                reader = VideoInput(recording_id)
                post_dir = os.path.splitext(recording_id)[0]
                os.makedirs(post_dir, exist_ok=True)
            else:
                loader = lambda d: holoframe.load(d)
                raw_dir = os.path.join(RECORDING_RAW_DIR, recording_id)
                post_dir = os.path.join(RECORDING_POST_DIR, recording_id)
                print("raw directory:", raw_dir)
                reader = RawReader(os.path.join(raw_dir, 'main'))
                raw_writer = RawWriter(self.session.out_sids, raw_dir)
                exits.enter_context(raw_writer)
            json_writers = {
                s: JsonWriter(s, post_dir)
                for s in self.session.out_sids
            }
            for w in json_writers.values():
                exits.enter_context(w)
            exits.enter_context(reader)

            for ts, d in reader:
                if isinstance(ts, float):
                    ts = ptgctl.util.format_epoch_time(ts)
                out = self.session.on_image(**loader(d))
                if raw_writer:
                    raw_writer.write(jsondump(out), ts)
                for k, x in out.items():
                    json_writers[k].write(jsondump({'data': x, 'timestamp': ts}), ts)



def noconfliict_ts(ts):
    return ts.split('-')[0] + '-*'


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(StepsApp)

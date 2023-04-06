import os
import orjson
import asyncio
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

from ptgprocess.omnimix import Omnivore, AudioSlowFast, OmniGRU

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
    def __init__(self, model, vocab, id_vocab, step_mask, prefix=None):
        self.model = model
        self.rgb_q = deque(maxlen=32)
        self.aud_q = deque(maxlen=16)
        self.rgb_emb_q = deque(maxlen=8 * self.emb_stride)
        self.aud_emb_q = deque(maxlen=8 * self.emb_stride)
        self.step_vocab = vocab
        self.step_id_vocab = id_vocab
        self.step_mask = torch.Tensor(step_mask).int()
        self.sim_decay = 0
        prefix = f'{prefix or ""}omnimix'

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
        x = torch.zeros(1, 3, 32, 224, 224).to(device)
        _, emb_vid = self.model.rgb(x, return_embedding=True)
        self.empty_rgb_emb = emb_vid#torch.zeros(1, 1024).to(device)
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
        return {
            self.step_sid: dict(zip(self.step_vocab.tolist(), steps.detach().cpu().tolist())) if steps is not None else None, 
            self.step_id_sid: dict(zip(self.step_id_vocab.tolist(), steps.detach().cpu().tolist())) if steps is not None else None, 
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
        print(aud.shape)
        aud = aud[-int(1.999*self.audio_sr):]
        print(aud.shape, aud.min(), aud.max())
        spec = self.model.aud.prepare_audio(aud, self.audio_sr)
        spec = [x.to(device) for x in spec]
        (verb_aud, noun_aud), emb_aud = self.model.aud(spec, return_embedding=True)
        print(emb_aud.shape)
        self.aud_emb_q.append(emb_aud)
        return verb_aud, noun_aud
    
    def _predict_step(self):
        vq = list(self.rgb_emb_q)[::self.emb_stride]
        aq = list(self.aud_emb_q)[::self.emb_stride]
        n_len = 8#max(len(vq), len(aq), 2)
        emb_rgbs = torch.stack([self.empty_rgb_emb]*(n_len - len(vq)) + list(vq), dim=1).to(device)
        emb_auds = torch.stack([self.empty_aud_emb]*(n_len - len(aq)) + list(aq), dim=1).to(device)
        steps, _ = self.model.mix(emb_rgbs, emb_auds)
        steps = F.softmax(steps[0, self.step_mask], dim=-1)
        return steps



class Omnimix(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = Omnivore()
        self.aud = AudioSlowFast()
        self.mix = OmniGRU()

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



class RecipeExit(Exception):
    pass


# step_ids = np.array([
#     'start',    
#     'end',    
#     "Place tortilla on cutting board.",     
#     "Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla",# leaving 1/2-inch", #uncovered at the edges.",     
#     "Clean the knife by wiping with a paper towel.",     
#     "Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.",     
#     "Clean the knife by wiping with a paper towel.",     
#     "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick.",# Roll it tight",# enough to prevent gaps, but not so tight that the filling leaks.",     
#     "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",     
#     "Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin",# between the last",# toothpick and the end of the roll. Discard ends.",     
#     "Slide floss under the tortilla, perpendicular to the length of the roll.",# Place the floss halfway", #between two toothpicks.",     
#     "Cross the two ends of the floss over the top of the tortilla roll. Holding",# one end of the floss in", #each hand, pull the floss ends in opposite directions to slice.",     
#     "Continue slicing with floss to create 5 pinwheels.",     
#     "Place the pinwheels on a plate.",    
#     "Measure 12 ounces of cold water and transfer to a kettle.",     
#     "Assemble the filter cone.  Place the dripper on top of a coffee mug.",     
#     "Prepare the filter insert by folding the paper filter in half to create a semi-circle",# and in half", #again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone.",     
#     "Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand",#, about 20 seconds. Transfer the grounds to the filter cone.",     
#     "Check the temperature of the water.",     
#     "Pour a small amount of water in the filter to wet the grounds. Wait about 30 seconds.",     
#     "Slowly pour the rest of the water over the grounds in a circular motion. Do not overfill", #beyond the top of the paper filter.",     
#     "Let the coffee drain completely into the mug before removing the dripper. Discard the paper",# filter and coffee grounds.",    
#     "Place the paper cupcake liner inside the mug. Set aside.",     
#     "Measure and add the flour, sugar, baking powder, and salt to the mixing bowl.",     
#     "Whisk to combine.",     
#     "Measure and add the oil, water, and vanilla to the bowl.",     
#     "Whisk batter until no lumps remain.",     
#     "Pour batter into prepared mug.",     
#     "Microwave the mug and batter on high power for 60 seconds.",     
#     "Check if the cake is done by inserting and toothpick into the center of the cake and then",# removing. If wet batter clings to the toothpick, microwave for an additional 5 seconds. If the toothpick comes out clean, continue.",     
#     "Invert the mug to release the cake onto a plate. Allow to cool until it is no longer hot",# to the touch, then carefully remove paper liner.",     
#     "While the cake is cooling, prepare to pipe the frosting. Scoop 4 spoonfuls of chocolate", #frosting into a zip-top bag and seal, removing as much air as possible.",     
#     "Use scissors to cut one corner from the bag to create a small opening 1/4-inch in diameter.",     
#     "Squeeze the frosting through the opening to apply small dollops of frosting to the plate",# in a circle around the base of the cake."
# ])
step_named_ids = np.array([
    '|start',    
    '|end',    

    # pinwheels
    "1|place tortilla",
    "2|scoop and spread nut butter",
    "3|clean knife",
    "4|scoop and spread jelly",
    "5|clean knife",
    "6|roll tortilla",
    "7|insert toothpicks",
    "8|trim tortilla ends",
    "9|slide floss underneath tortilla",
    "10|slice tortilla with floss",
    "11|slice tortilla into 5 pinwheels",
    "12|place pinwheels on plate",
    # "Use a butter knife to scoop nut butter from the jar. Spread nut butter onto tortilla",# leaving 1/2-inch", #uncovered at the edges.",     
    # "Clean the knife by wiping with a paper towel.",     
    # "Use the knife to scoop jelly from the jar. Spread jelly over the nut butter.",     
    # "Clean the knife by wiping with a paper towel.",     
    # "Roll the tortilla from one end to the other into a log shape, about 1.5 inches thick.",# Roll it tight",# enough to prevent gaps, but not so tight that the filling leaks.",     
    # "Secure the rolled tortilla by inserting 5 toothpicks about 1 inch apart.",     
    # "Trim the ends of the tortilla roll with the butter knife, leaving 1⁄2 inch margin",# between the last",# toothpick and the end of the roll. Discard ends.",     
    # "Slide floss under the tortilla, perpendicular to the length of the roll.",# Place the floss halfway", #between two toothpicks.",     
    # "Cross the two ends of the floss over the top of the tortilla roll. Holding",# one end of the floss in", #each hand, pull the floss ends in opposite directions to slice.",     
    # "Continue slicing with floss to create 5 pinwheels.",     
    # "Place the pinwheels on a plate.",  

    # coffee 
    "1|measure water",
    "2|place coffee dripper on mug", 
    "3|fold filter into quarters", 
    "4|weigh coffee beans and grind",
    "5|measure water temperature",
    "6|pour water onto grounds",
    "7|slowly pour rest of water",
    "8|let water drain then remove dripper",
    # "Measure 12 ounces of cold water and transfer to a kettle.",     
    # "Assemble the filter cone.  Place the dripper on top of a coffee mug.",     
    # "Prepare the filter insert by folding the paper filter in half to create a semi-circle",# and in half", #again to create a quarter-circle. Place the paper filter in the dripper and spread open to create a cone.",     
    # "Weigh the coffee beans and grind until the coffee grounds are the consistency of coarse sand",#, about 20 seconds. Transfer the grounds to the filter cone.",     
    # "Check the temperature of the water.",     
    # "Pour a small amount of water in the filter to wet the grounds. Wait about 30 seconds.",     
    # "Slowly pour the rest of the water over the grounds in a circular motion. Do not overfill", #beyond the top of the paper filter.",     
    # "Let the coffee drain completely into the mug before removing the dripper. Discard the paper",# filter and coffee grounds.",    

    "1|place cupcake liner in mug",
    "2|measure dry ingredients",
    "3|whisk to combine", 
    "4|add oil, water, vanilla", 
    "5|whisk batter until smooth", 
    "6|pour batter into mug",
    "7|microwave mug for 60s", 
    "8|check cake with toothpick", 
    "9|invert mug to release cake", 
    "10|prepare frosting bag", 
    "11|cut frosting bag corner", 
    "12|apply frosting to plate", 
    # "Place the paper cupcake liner inside the mug. Set aside.",    
    # "Measure and add the flour, sugar, baking powder, and salt to the mixing bowl.",     
    # "Whisk to combine.",     
    # "Measure and add the oil, water, and vanilla to the bowl.",     
    # "Whisk batter until no lumps remain.",     
    # "Pour batter into prepared mug.",     
    # "Microwave the mug and batter on high power for 60 seconds.",     
    # "Check if the cake is done by inserting and toothpick into the center of the cake and then",# removing. If wet batter clings to the toothpick, microwave for an additional 5 seconds. If the toothpick comes out clean, continue.",     
    # "Invert the mug to release the cake onto a plate. Allow to cool until it is no longer hot",# to the touch, then carefully remove paper liner.",     
    # "While the cake is cooling, prepare to pipe the frosting. Scoop 4 spoonfuls of chocolate", #frosting into a zip-top bag and seal, removing as much air as possible.",     
    # "Use scissors to cut one corner from the bag to create a small opening 1/4-inch in diameter.",     
    # "Squeeze the frosting through the opening to apply small dollops of frosting to the plate",# in a circle around the base of the cake."
    # tourniquet
    "1|place tourniquet above wound",
    "2|pull tourniquet tight",
    "3|apply strap",
    "4|turn windlass until tight",
    "5|lock windlass into keeper",
    "6|pull strap over windlass keeper",
    "7|secure strap and windlass",
    "8|mark time on strap with marker",
    # 34:"Place tourniquet over affected extremity 2-3 inches above wound site. ",
    # 35:"Pull tourniquet tight.",
    # 36:"Apply strap to strap body.",
    # 37:"Turn windless clock wise or counter clockwise until hemorrhage is controlled.",
    # 38:"Lock windless into the windless keeper.",
    # 39:"Pull remaining strap over the windless keeper.",
    # 40:"Secure strap and windless keeper with keeper securing device.",
    # 41:"Mark time on securing device strap with permanent marker.",
])

step_ids = np.array([
    "-2",#'|start',    
    "-1",#'|end',    

    # pinwheels
    "0",#"1|place tortilla",
    "1",#"2|scoop and spread nut butter",
    "2",#"3|clean knife",
    "3",#"4|scoop and spread jelly",
    "4",#"5|clean knife",
    "5",#"6|roll tortilla",
    "6",#"7|insert toothpicks",
    "7",#"8|trim tortilla ends",
    "8",#"9|slide floss underneath tortilla",
    "9",#"10|slice tortilla with floss",
    "10",#"11|slice tortilla into 5 pinwheels",
    "11",#"12|place pinwheels on plate",

    # coffee 
    "0",#"1|measure water",
    "1",#"2|place coffee dripper on mug", 
    "2",#"3|fold filter into quarters", 
    "3",#"4|weigh coffee beans and grind",
    "4",#"5|measure water temperature",
    "5",#"6|pour water onto grounds",
    "6",#"7|slowly pour rest of water",
    "7",#"8|let water drain then remove dripper",
    
    # mugcake
    "0",#"1|place cupcake liner in mug",
    "1",#"2|measure dry ingredients",
    "2",#"3|whisk to combine", 
    "3",#"4|add oil, water, vanilla", 
    "4",#"5|whisk batter until smooth", 
    "5",#"6|pour batter into mug",
    "6",#"7|microwave mug for 60s", 
    "7",#"8|check cake with toothpick", 
    "8",#"9|invert mug to release cake", 
    "9",#"10|prepare frosting bag", 
    "10",#"11|cut frosting bag corner", 
    "11",#"12|apply frosting to plate", 

    # tourniquet
    "0",#"1|place tourniquet above wound",
    "1",#"2|pull tourniquet tight",
    "2",#"3|apply strap",
    "3",#"4|turn windlass until tight",
    "4",#"5|lock windlass into keeper",
    "5",#"6|pull strap over windlass keeper",
    "6",#"7|secure strap and windlass",
    "7",#"8|mark time on strap with marker",
])
common_step_mask = np.array([0, 1])
recipe_step_mask = {
    'pinwheels':  np.concatenate([np.arange(2, 14)]).astype(int),
    'coffee':     np.concatenate([np.arange(14, 22)]).astype(int),
    'mugcake':    np.concatenate([np.arange(22, 34)]).astype(int),
    'tourniquet': np.concatenate([np.arange(34, 42)]).astype(int),
}


class StepsApp:
    vocab_key = 'steps_simple'
    decay = 0.5

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'omnimix',
                              password=os.getenv('API_PASS') or 'omnimix')
        self.model = Omnimix().cuda()
        self.model.eval()  # LOL fuck me. forgetting this is a bad time

    @torch.no_grad()
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
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
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

    def start_session(self, recipe_id, prefix=None):
        '''Initialize the action session for a recipe.'''
        if not recipe_id:
            raise RecipeExit("no recipe set.")
        # recipe = self.api.recipes.get(recipe_id) or {}
        # vocab = recipe.get(self.vocab_key)
        # step_map = recipe.get('step_map') or {}
        # if not vocab:
        #     raise Exception(f"\nRecipe {recipe_id}.{self.vocab_key} returned no steps in {set(recipe)}.")
        if recipe_id not in recipe_step_mask:
            raise RecipeExit(f"{recipe_id} not supported by this model.")
        step_mask = recipe_step_mask[recipe_id]
        vocab = step_named_ids[step_mask]
        id_vocab = step_ids[step_mask]
        self.session = StepsSession(self.model, vocab, id_vocab, step_mask, prefix=prefix)

    async def run_recipe(self, recipe_id, prefix=None):
        '''Run the recipe.'''
        self.start_session(recipe_id, prefix=prefix)

        # stream ids
        # in_rgb_sid = 'main'
        # in_aud_sid = 'mic0'
        in_rgb_sid = f'{prefix or ""}main'
        in_aud_sid = f'{prefix or ""}mic0'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        print(self.session.out_sids)

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([in_rgb_sid, in_aud_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(self.session.out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            # self.start_session(recipe_id, prefix=prefix)
                            return

                        # predict actions
                        preds = None
                        if sid == in_rgb_sid:
                            preds = self.session.on_image(**holoframe.load(d))
                        elif sid == in_aud_sid:
                            preds = self.session.on_audio(**holoframe.load(d))
                        if preds is not None:
                            preds = {k: v for k, v in preds.items() if v is not None}
                            await ws_push.send_data(
                                [jsondump({k: round(v, 2) for k, v in sorted(x.items(), key=lambda x: -x[1])}) for x in preds.values()], 
                                list(preds.keys()), 
                                #[noconfliict_ts(t)]*len(preds)
                            )

    @torch.no_grad()
    def run_offline(self, recording_id, recipe_id, prefix=None):
        from contextlib import ExitStack
        from ptgprocess.util import VideoInput
        from ptgprocess.record import RawReader, RawWriter, JsonWriter
        self.start_session(recipe_id, prefix=prefix)

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

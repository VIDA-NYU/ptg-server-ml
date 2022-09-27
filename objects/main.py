import os
import orjson
from collections import OrderedDict

import asyncio
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np

# import deep_sort

import ptgctl
import ptgctl.util
from ptgctl import holoframe
from ptgctl import pt3d
from ptgprocess.detic import Detic

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

DEFAULT_VOCAB = 'lvis'

class _SliceView:
    __slots__ = 'method'
    def __init__(self, method):
        self.method = method
    def __getitem__(self, k):
        if isinstance(k, slice):
            return list(self.method(k))
        return [k]

class dicque(OrderedDict):
    '''Like a deque but with dicts! lol'''
    def __init__(self, *a, maxlen=0, **kw):
        self._max = maxlen
        super().__init__(*a, **kw)

    def __getitem__(self, k):
        gi=super().__getitem__
        if isinstance(k, slice):
            return [gi(k) for k in self._key_islice(k)]
        return gi(k)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if len(self) > self._max > 0:
            self.popitem(False)

    def closest(self, tms):
        return min(self, key=lambda t: abs(tms-t), default=None)

    def first(self, *a):
        return next(iter(self), *a)

    def last(self, *a):
        return next(reversed(self), *a)

    @property
    def ks(self):
        return _SliceView(self._key_islice)

    def _key_islice(self, slc):
        it = iter(self) if (slc.step or 1) >= 0 else reversed(self)
        start = slc.start
        stop = slc.stop
        if start is not None:
            # ideally we'd like to jump the linked list
            for k in it:
                if k >= start:
                    yield k
                    break
        if stop is None:
            yield from it
            return

        for k in it:
            if k > stop:
                return
            yield k




class ObjectsApp:
    
    
    def __init__(self, hand_decay=0.5, max_depth_time_diff=1, max_depth_dist=7, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'detic',
                              password=os.getenv('API_PASS') or 'detic')
        self.model = Detic(device='cuda:0', **kw)
        self.hand_decay = hand_decay
        self.max_depth_time_diff = max_depth_time_diff
        self.max_depth_dist = max_depth_dist
        self.vocab_keys = ['objects']
        self.EXTRA_VOCAB = ['person', 'feet']

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
        recipe_sid = 'event:recipe:id'
        in_sids = ['main', 'depthlt', 'depthltCal']

        self.mean_person_confs_decay = 0

        # create queues for each data stream
        self.diqs = {sid: dicque() for sid in in_sids}
        diqmain, diqdepth, diqdepthcal = [self.diqs[s] for s in in_sids]

        # get the current recipe at the start
        self.change_recipe(self.api.sessions.current_recipe())

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect(in_sids + [recipe_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect('*', batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    xs = await ws_pull.recv_data()
                    if xs:
                        pbar.update()
                    # first, load and record all data so we can access any of it from their queues
                    for sid, t, x in xs:
                        pbar.set_description(f'{sid} {t}')
                        # watch for recipe changes
                        if sid == recipe_sid:
                            self.change_recipe(x)
                            continue

                        tms = int(t.split('-')[0])
                        self.diqs[sid][tms] = _d = holoframe.load(x)
                        _d['_t'] = t

                    # then loop through and use the new data
                    for sid, t, x in xs:
                        pbar.set_description(f'{sid} {t}')
                        tms = int(t.split('-')[0])

                        if sid == 'main':
                            # compute object locations in 2D

                            # predict objects
                            main: dict = diqmain[tms]
                            im = main['image']
                            xyxy, confs, class_ids, labels = self.predict(im)
                            # normalize box
                            h, w = im.shape[:2]
                            xyxyn = xyxy.copy()
                            xyxyn[:,[0,2]] /= w
                            xyxyn[:,[1,3]] /= h 

                            objs = self.zip_objs(
                                ['xyxyn', 'confidence', 'class_id', 'label'],
                                [xyxyn, confs, class_ids, labels])

                            # send 2d box - TODO: parallelize
                            await ws_push.send_data([
                                jsondump(objs),
                                jsondump(self.detect_hands(labels, confs)),
                            ], ['detic:image', 'detic:hands'], [t, t])


                            # convert 2d locations to 3d

                            # get the points object
                            t_depth = diqdepth.closest(tms)
                            if not t_depth or abs(t_depth - tms) > self.max_depth_time_diff:
                                continue
                            pts3d = self.get_pts3d(tms, t_depth, diqdepthcal.last())

                            # compute the 3d location in world space
                            xyz_center, dist = pts3d.transform_center_withinbbox(xyxy)
                            valid = dist < self.max_depth_dist  # make sure the points aren't too far
                            log.debug('%d/%d boxes valid. dist in [%f,%f]', valid.sum(), len(valid), dist.min(initial=np.inf), dist.max(initial=0))

                            # add 3d position to objects
                            for obj, xyz_center, valid, dist in zip(objs, xyz_center, valid, dist):
                                obj['xyz_center'] = xyz_center
                                obj['depth_map_dist'] = dist

                            # TODO: object tracking ... and add tracking info to 
                            
                            # send 3d boxes
                            await ws_push.send_data([jsondump(objs)], ['detic:world'], [t])
                        

                        
                        if sid == 'depthlt':
                            # get the points object
                            t_main = self.diqs['main'].closest(tms)
                            if not t_main or abs(t_main - tms) > self.max_depth_time_diff:
                                continue
                            pts3d = self.get_pts3d(t_main, tms, diqdepthcal.last())
                            
                            # write point cloud
                            await ws_push.send_data([
                                jsondump({
                                    'time': t,
                                    'color': (pts3d.rgb * 255).astype('uint8'),
                                    'xyz_world': pts3d.xyz_depth_world,
                                }),
                            ], ['pointcloud'], [t])


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

    def detect_hands(self, labels, confs):
        is_person = labels == 'person'
        person_confs = confs[is_person] if is_person.sum() else np.zeros((1,))
        mean_person_confs = np.nanmean(person_confs)
        self.mean_person_confs_decay = (
            (1 - self.hand_decay) * mean_person_confs + 
            self.hand_decay * self.mean_person_confs_decay)
        return {
            'n_person_boxes': is_person.sum(),
            'max_conf': np.max(person_confs, initial=0),
            'mean_conf': mean_person_confs,
            'mean_conf_smooth': self.mean_person_confs_decay,
        }

    def get_pts3d(self, tmain: int, tdepthlt: int, tdepthltCal: int): # TODO: lru_cache(maxsize=8)
        main: dict = self.diqs['main'][tmain]
        depthlt: dict = self.diqs['depthlt'][tdepthlt]
        depthltCal: dict = self.diqs['depthltCal'][tdepthltCal]
        return pt3d.Points3D(
            main['image'], depthlt['image'], depthltCal['lut'],
            depthlt['rig2world'], depthltCal['rig2cam'], main['cam2world'],
            [main['focalX'], main['focalY']], [main['principalX'], main['principalY']],
            generate_point_cloud=True)

    def zip_objs(self, keys, xs):
        return [dict(zip(keys, xs)) for xs in zip(*xs)]


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


if __name__ == '__main__':
    import fire
    fire.Fire(ObjectsApp)

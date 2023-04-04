import os
import numpy as np
import utils
import impl
import orjson
import ptgctl
import ptgctl.util
from ptgctl import holoframe
from collections import OrderedDict


class dicque(OrderedDict):
    def __init__(self, *a, maxlen=0, **kw):
        self._max = maxlen
        super().__init__(*a, **kw)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if self._max > 0 and len(self) > self._max:
            self.popitem(False)

    def closest(self, tms):
        k = min(self, key=lambda t: abs(tms-t), default=None)
        return k


class MemoryApp:
    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'memory',
                              password=os.getenv('API_PASS') or 'memory')

    @ptgctl.util.async2sync
    async def run(self, prefix=None):
        data = {}

        in_sids = ['main', 'depthlt', 'depthltCal']
        obj_sid = 'detic:image:for3d'
        RECIPE_SID = 'event:recipe:id'
        output_sid = 'detic:memory'

        mem = impl.Memory()
        hist_d = dicque(maxlen=20)
        hist_rgb = dicque(maxlen=20)
        server_time_to_sensor_time = dicque(maxlen=20)

        async with self.api.data_pull_connect(in_sids + [obj_sid] + [RECIPE_SID]) as ws_pull, \
                self.api.data_push_connect(output_sid, batch=True) as ws_push:

            data.update(holoframe.load_all(self.api.data('depthltCal')))
            while True:
                for sid, t, buffer in await ws_pull.recv_data():
                    tms = int(t.split('-')[0])

                    if sid == RECIPE_SID:
                        mem = impl.Memory()
                        hist_d = dicque(maxlen=20)
                        hist_rgb = dicque(maxlen=20)
                        server_time_to_sensor_time = dicque(maxlen=20)
                        print("memory cleared")
                        continue

                    # detic projection
                    if sid == obj_sid:
                        if not hist_d:
                            continue
                        if tms not in server_time_to_sensor_time:
                            continue
                        sensor_time = server_time_to_sensor_time[tms]
                        depth_time = hist_d.closest(sensor_time)
                        depth_frame = hist_d[depth_time]
                        rgb_frame = hist_rgb[sensor_time]
                        detic_result = orjson.loads(buffer)

                        height, width = rgb_frame['image'].shape[:2]

                        depth_points = utils.get_points_in_cam_space(
                            depth_frame['image'], data['depthltCal']['lut'])
                        xyz, _ = utils.cam2world(
                            depth_points, data['depthltCal']['rig2cam'], depth_frame['rig2world'])
                        pos_image, mask = utils.project_on_pv(
                            xyz, rgb_frame['image'], rgb_frame['cam2world'],
                            [rgb_frame['focalX'], rgb_frame['focalY']], [rgb_frame['principalX'], rgb_frame['principalY']])

                        nms_idx = impl.nms(
                            detic_result["objects"], rgb_frame['image'].shape[:2])
                        detections = []
                        for i in nms_idx:
                            o = detic_result["objects"][i]
                            label, xyxyn = o["label"], o["xyxyn"]
                            if label in {'person'}:
                                continue

                            y1, y2, x1, x2 = int(xyxyn[1]*height), int(xyxyn[3]*height), int(
                                xyxyn[0]*width), int(xyxyn[2]*width)
                            pos_obj = pos_image[y1:y2, x1:x2, :]
                            mask_obj = mask[y1:y2, x1:x2]
                            pos_obj = pos_obj[mask_obj]
                            if pos_obj.shape[0] == 0:
                                continue
                            pos_obj = pos_obj.mean(axis=0)

                            detections.append(
                                impl.PredictionEntry(pos_obj, label, o["confidence"]))

                        intrinsic_matrix = np.array([[rgb_frame['focalX'], 0, width-rgb_frame['principalX']], [
                            0, rgb_frame['focalY'], rgb_frame['principalY']], [0, 0, 1]])
                        mem.update(detections, 1, sensor_time,
                                   intrinsic_matrix, np.linalg.inv(rgb_frame['cam2world']), rgb_frame['image'].shape[:2])

                        await ws_push.send_data([orjson.dumps(mem.to_json(), option=orjson.OPT_SERIALIZE_NUMPY)])
                        continue

                    # store the data
                    data[sid] = d = holoframe.load(buffer)
                    d['_tms'] = tms

                    if sid == 'depthlt':
                        hist_d[d['time']] = d
                    elif sid == 'main':
                        hist_rgb[d['time']] = d
                        server_time_to_sensor_time[tms] = d['time']


if __name__ == '__main__':
    import fire
    fire.Fire(MemoryApp)

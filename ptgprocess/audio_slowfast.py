import os
import sys
import numpy as np
import librosa
import torch
from torch import nn

asf_path = os.path.join(os.path.dirname(__file__), 'auditory-slow-fast')
sys.path.insert(0,  asf_path)

import audio_slowfast
import audio_slowfast.utils.checkpoint as cu
from audio_slowfast.models import build_model
from audio_slowfast.config.defaults import get_cfg


MODEL_DIR = os.getenv('MODEL_DIR') or 'models'

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'auditory-slow-fast')
DEFAULT_CONFIG = os.path.join(CONFIG_DIR, 'configs/EPIC-KITCHENS/SLOWFAST_R50.yaml')
DEFAULT_MODEL = os.path.join(MODEL_DIR, 'SLOWFAST_EPIC.pyth')

class AudioSlowFast(nn.Module):
    eps = 1e-6
    def __init__(self, checkpoint=DEFAULT_MODEL, cfg_file=DEFAULT_CONFIG):
        super().__init__()
        # init config
        self.cfg = cfg = get_cfg()
        cfg.merge_from_file(cfg_file)

        # get vocab classes
        self.vocab = []
        if cfg.MODEL.VOCAB_FILE:
            import json
            cfg.MODEL.VOCAB_FILE = os.path.join(os.path.dirname(cfg_file), cfg.MODEL.VOCAB_FILE)
            self.vocab = json.load(open(cfg.MODEL.VOCAB_FILE))

        # window params
        window_size = cfg.AUDIO_DATA.WINDOW_LENGTH
        step_size = cfg.AUDIO_DATA.HOP_LENGTH
        self.win_size = int(round(window_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
        self.hop_size = int(round(step_size * cfg.AUDIO_DATA.SAMPLING_RATE / 1e3))
        self.num_frames = cfg.AUDIO_DATA.NUM_FRAMES
        self.num_classes = cfg.MODEL.NUM_CLASSES

        # build and load model
        cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint
        cfg.NUM_GPUS = min(cfg.NUM_GPUS, torch.cuda.device_count())
        self.model = build_model(cfg)
        cu.load_test_checkpoint(cfg, self.model)
        self.model.head.__class__ = ResNetBasicHead

    def prepare_audio(self, y, sr):
        spec = librosa.stft(
            y, n_fft=2048,
            window='hann',
            hop_length=self.hop_size,
            win_length=self.win_size,
            pad_mode='constant')
        mel_basis = librosa.filters.mel(
            sr=sr, n_fft=2048, n_mels=128, htk=True, norm=None)
        spec = np.dot(mel_basis, np.abs(spec))
        spec = np.log(spec + self.eps)
        npad = max(0, self.num_frames - spec.shape[-1])
        # print(spec.shape)
        spec = np.pad(spec, ((0, npad), (0, 0)), 'edge')
        # print(spec.shape)
        spec = librosa.util.frame(spec, frame_length=400, hop_length=100)
        # print(spec.shape)
        spec = spec.transpose((2, 1, 0))[:,None]

        # print(spec.shape)
        spec = torch.tensor(spec, dtype=torch.float) # mono to stereo
        spec = pack_pathway_output(self.cfg, spec)
        # print(spec.shape)
        return spec

    def forward(self, specs, return_embedding=False):
        z = self.model(specs)
        y = self.model.head.project_verb_noun(z)
        if self.model.training:
            y = [x.view((len(x), -1, s)) for x, s in zip(y, self.num_classes)]
        if return_embedding:
            return y, z
        return y


class ResNetBasicHead(audio_slowfast.models.head_helper.ResNetBasicHead):
    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H) -> (N, T, H, C).
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x

    def _proj(self, x, proj):
        x_v = proj(x)
        # Performs fully convlutional inference.
        if not self.training:
            x_v = self.act(x_v)
            x_v = x_v.mean([1, 2])
        return x_v.view(x_v.shape[0], -1)

    def project_verb_noun(self, x):
        if isinstance(self.num_classes, (list, tuple)):
            return self._proj(x, self.projection_verb), self._proj(x, self.projection_noun)
        return self._proj(x, self.projection)



def pack_pathway_output(cfg, spec):
    """[ch X time X freq] -> [ [ch X slow time X freq], [ch X fast time X freq] ]"""
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        return [spec]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        # Perform temporal sampling from the fast pathway.
        T = spec.shape[-2]
        i = torch.linspace(0, T - 1, T // cfg.SLOWFAST.ALPHA).long()
        slow = torch.index_select(spec, -2, i)
        return [slow, spec]

    raise NotImplementedError(
        f"Model arch {cfg.MODEL.ARCH} is not in {cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(path):
    model = AudioSlowFast()
    model.eval()
    vocab_verb, vocab_noun = model.vocab
    y, sr = librosa.load(path, sr=None)
    spec = model.prepare_audio(y, sr)
    
    print([x.shape for x in spec])
    verb, noun = model([x.to(device) for x in spec])
    print(verb.shape, noun.shape)
    i_vs, i_ns = torch.argmax(verb, dim=-1), torch.argmax(noun, dim=-1)
    for v, n, i_v, i_n in zip(verb, noun, i_vs, i_ns):
        print(f'{vocab_verb[i_v]:>12}: ({v[i_v]:.2%})  {vocab_noun[i_n]:>12}: ({n[i_n]:.2%})')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
import os
import sys
import numpy as np
import librosa
import torch
from torch import nn

import gdown

mod_path = os.path.join(os.path.dirname(__file__), 'procedural_step_recog')
sys.path.insert(0,  mod_path)

from step_recog.config.defaults import get_cfg

# from .omnivore import Omnivore
# from .audio_slowfast import AudioSlowFast

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = os.getenv('MODEL_DIR') or 'models'
DEFAULT_CONFIG = os.path.join(mod_path, 'config/STEPGRU.yaml')
DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, 'model_best.pt')
DEFAULT_CHECKPOINT2 = os.path.join(MODEL_DIR, 'model_best_multimodal.pt')

if not os.path.isfile(DEFAULT_CHECKPOINT2):
    gdown.download(id="1ArZFX4LuuB4SbmmWSDit4S8gPhkc2DRq", output=DEFAULT_CHECKPOINT2)

# class Omnimix(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.rgb = Omnivore(return_embedding=True)
#         self.aud = AudioSlowFast(return_embedding=True)
#         self.mix = OmniGRU()

#     def forward(self, im, aud):
#         action_rgb, emb_rgb = self.rgb(im)
#         verb_rgb, noun_rgb = self.rgb.project_verb_noun(action_rgb)

#         (verb_aud, noun_aud), emb_aud = self.aud(aud)
#         step = self.mix(emb_rgb, emb_aud)
#         return step, action_rgb, (verb_rgb, noun_rgb), (verb_aud, noun_aud)



class OmniGRU2(nn.Module):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT2, cfg_file=DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        rgb_size = 1024
        audio_size = 2304
        input_dim = rgb_size
        hidden_dim = cfg.MODEL.HIDDEN_SIZE
        output_dim = cfg.MODEL.OUTPUT_DIM
        n_layers = 2
        drop_prob = 0.2

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rgb_fc = nn.Linear(rgb_size, int(input_dim/2))
        self.obj_fc = nn.Linear(512, int(input_dim/2))
        self.obj_proj = nn.Linear(517, int(input_dim/2))
        self.frame_proj = nn.Linear(517, int(input_dim/2))
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim + 2)
        self.relu = nn.ReLU()

        self.load_state_dict(torch.load(checkpoint))

    def forward(self, x, h):

        omni, objs, frame = x
        omni_in = self.relu(self.rgb_fc(x[0]))

        obj_proj = self.relu(self.obj_proj(x[1]))
        frame_proj = self.relu(self.frame_proj(x[2]))
        values = torch.softmax(torch.sum(frame_proj*obj_proj,dim=-1,keepdims=True),dim=-2)
        obj_in = torch.sum(obj_proj*values,dim=-2)
        obj_in = self.relu(self.obj_fc(obj_in))
        
        x = torch.concat((omni_in,obj_in),-1)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h


class OmniGRU(nn.Module):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, cfg_file=DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        rgb_size = 1024
        audio_size = 2304
        input_dim = rgb_size
        hidden_dim = cfg.MODEL.HIDDEN_SIZE
        output_dim = cfg.MODEL.OUTPUT_DIM
        n_layers = 2
        drop_prob = 0.2

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.rgb_fc = nn.Linear(rgb_size, int(input_dim/2))
        self.audio_fc = nn.Linear(audio_size, int(input_dim/2))
        self.rgb_bn = nn.BatchNorm1d(int(input_dim/2))
        self.aud_bn = nn.BatchNorm1d(int(input_dim/2))
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.init_hidden(32)

        self.load_state_dict(torch.load(checkpoint))
        
    def forward(self, x_rgb, x_aud, h=None):
        x_rgb = self.rgb_fc(x_rgb)
        x_rgb = self.rgb_bn(x_rgb.transpose(1, 2)).transpose(1, 2)
        x_rgb = self.relu(x_rgb)

        x_aud = self.audio_fc(x_aud)
        x_aud = self.aud_bn(x_aud.transpose(1, 2)).transpose(1, 2)
        x_aud = self.relu(x_aud)
        
        x = torch.concat((x_rgb, x_aud), -1)
        x, h = self.gru(x, h)
        x = self.relu(x[:, -1])
        x = self.fc(x)
        return x, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

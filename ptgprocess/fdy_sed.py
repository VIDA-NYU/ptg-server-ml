'''Monocular depth estimation

Source: https://github.com/nianticlabs/monodepth2

'''
import os
import numpy as np
import onnxruntime
import librosa

localfile = lambda *f: os.path.join(os.path.dirname(__file__), *f)

TEACHER_PATH = localfile('teacher.onnx')
STUDENT_PATH = localfile('student.onnx')

class Model:
    labels = [
        "Alarm_bell_ringing",
        "Blender",
        "Cat",
        "Dishes",
        "Dog",
        "Electric_shaver_toothbrush",
        "Frying",
        "Running_water",
        "Speech",
        "Vacuum_cleaner",
    ]
    def __init__(self, path):
        self.sess = sess = onnxruntime.InferenceSession(path)
        self.input_names = [i.name for i in sess.get_inputs()]

    def predict(self, *inputs):
        return self.sess.run(None, {k: np.asarray(x) for k, x in zip(self.input_names, inputs)})


class FDYSED(Model):
    def __init__(self, path=None, teacher=False):
        super().__init__(path or (TEACHER_PATH if teacher else STUDENT_PATH))
        self.sr = 16000
        self.hop_length = 256
        self.hop_secs = self.hop_length / self.sr

    def preprocess(self, y, sr):
        # norm audio
        y = y[0,:] if y.ndim > 1 else y
        y = librosa.resample(y=y, orig_sr=sr, target_sr=self.sr)
        y = y / (np.max(np.abs(y)) + 1e-10)
        # get log mel spec
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, hop_length=self.hop_length, 
            power=1, htk=True, window='hamming', norm=None)
        S = librosa.amplitude_to_db(S, amin=np.sqrt(1e-5), top_db=None)
        # clip and normalize
        S = np.clip(S, -50, 80)
        S = S[None]
        low, high = S.min(axis=(0, 2), keepdims=True), S.max(axis=(0, 2), keepdims=True)
        S = (S - low) / (high - low)
        return S

def main(*paths):
    student = FDYSED()
    teacher = FDYSED(teacher=True)
    for f in paths:
        y, sr = librosa.load(f, sr=None, mono=False)
        mels = student.preprocess(y, sr)
        print(mels.shape)
        student_strong, student_weak = student.predict(mels)
        teacher_strong, teacher_weak = teacher.predict(mels)
        print('Student:')
        print_labels(student_weak[0], student.labels)
        print('Teacher:')
        print_labels(teacher_weak[0], student.labels)
        plot_results(f, mels[0], student_strong[0], teacher_strong[0], student.labels, student.hop_secs)


def print_labels(preds, labels):
    '''Print out the model predictions.'''
    for i, p in sorted(enumerate(preds), key=lambda x: x[1], reverse=True):
        print(p, '\t', labels[i])

def plot_results(path, logmels, stud_probs, tch_probs, labels, spec_hop_secs=1):
    '''Plot the model inputs / outputs.'''

    out_hop_secs = logmels.shape[1] / stud_probs.shape[1] * spec_hop_secs

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    plt.imshow(logmels, aspect='auto')
    N = logmels.shape[1]-1
    yticks = np.linspace(0, N, 10, dtype=int)
    plt.xticks(yticks, [f'{x:.1f}' for x in yticks*spec_hop_secs])

    plt.ylabel('mel spec')

    plt.subplot(312)
    plt.title('Student')
    plt.imshow(stud_probs, aspect='auto')

    N = stud_probs.shape[1]-1
    yticks = np.linspace(0, N, 10, dtype=int)
    plt.xticks(yticks, [f'{x:.1f}' for x in yticks*out_hop_secs])

    N = stud_probs.shape[0]-1
    yticks = np.linspace(0, N, N+1, dtype=int)
    plt.yticks(yticks, np.array([labels[i] for i in yticks]))

    plt.subplot(313)
    plt.title('Teacher')
    plt.imshow(tch_probs, aspect='auto')

    N = tch_probs.shape[1]-1
    yticks = np.linspace(0, N, 10, dtype=int)
    plt.xticks(yticks, [f'{x:.1f}' for x in yticks*out_hop_secs])

    N = tch_probs.shape[0]-1
    yticks = np.linspace(0, N, N+1, dtype=int)
    plt.yticks(yticks, np.array([labels[i] for i in yticks]))

    plt.ylabel('time')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dirname, fname = os.path.dirname(path), os.path.basename(path)
    plt.suptitle(fname)
    plt.savefig(os.path.join(dirname, f'plot_{fname}.png'))
    plt.close()



def debug_outputs(z):
    print(len(z))
    print([zi.shape for zi in z])
    return z

if __name__ == '__main__':
    import fire
    fire.Fire(main)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_action_df(annotations_file, video, include_no_action=True):
    ########################
    # pre-process the data #
    ########################

    # load dataframe
    df = pd.read_csv(annotations_file)

    # find the rows of the video
    df = df.loc[df['video_id']==video]

    # sort the actions in order
    df = df.sort_values('start_frame', ignore_index=True)

    # keep only the narration and the frame start/stop
    df = df[['narration','start_frame','stop_frame']]

    # find gaps between actions
    df['overlaps_previous'] = df.start_frame <= df.stop_frame.shift().fillna(-1)

    # create no action segments
    stop_frames = df.start_frame[~df.overlaps_previous]
    start_frames = df.stop_frame.shift().fillna(0).astype(int)[~df.overlaps_previous]
    noac_df = pd.DataFrame({
            'narration': 'no action',
            'start_frame': start_frames.tolist(),
            'stop_frame': stop_frames.tolist(),
    }, index=df.index[~df.overlaps_previous] - 0.5)
    df = (
        pd.concat([df, noac_df], axis=0, ignore_index=False)
          .sort_index()
          .drop(columns=['overlaps_previous'])
          .reset_index()
          .drop(columns=['index']))

    # check that all values are in good shape
    assert (np.diff(df.start_frame) >= 0).all()
    assert (df.stop_frame - df.start_frame > 0).all()

    # create a column with each action duration
    df['nframes'] = df.stop_frame - df.start_frame
    return df

def get_transition_matrix(df, actions, win_size=30, confusion=0.1):
    ############################
    # create dependency matrix #
    ############################

    # get the unique actions
    actions2i = {v:i for i,v in enumerate(actions)}

    # initialize empty matrix to store values in
    matrix = np.zeros((len(actions2i),len(actions2i)))

    for i,r in df.iterrows():
        matrix[actions2i[r['narration']],actions2i[r['narration']]] += r['nframes']/win_size
        if i < len(df)-1:
            nr = df.loc[i+1]
            matrix[actions2i[nr['narration']],actions2i[r['narration']]] += 1

    matrix /= np.max(matrix,axis=1,keepdims=True)
    matrix += confusion
    matrix = np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)
    return matrix


def wordiou(sentenceA, sentenceB):
    wA = set(sentenceA.split(' '))
    wB = set(sentenceB.split(' '))
    return len(wA & wB) / len(wA | wB)

def get_emission_matrix(actions, confusion=0.1):
    matrix = np.eye(len(actions))
    for i, ai in enumerate(actions):
        for j, aj in enumerate(actions[i+1:], i+1):
            matrix[i, j] = matrix[j, i] = wordiou(ai, aj)

    matrix += confusion
    matrix = np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)
    return matrix

def get_gaussian_features(actions, cov=0.25, margin=0.1):
    means = np.eye(len(actions))
    cov *= np.ones((len(actions), len(actions)))
    cov += np.random.randn(*cov.shape)*0.01
    cov = np.maximum(cov, 0.1)
    means += margin
    means = np.exp(means) / np.sum(np.exp(means), axis=1, keepdims=True)
    return means, cov

def probs_to_obs_counts(sim, n=100):
    counts = (sim * n).astype(int)
    print(counts.sum(-1).tolist())
    imax = np.argmax(counts, axis=-1)
    #ij = np.ix_(np.arange(len(counts)), imax)
    counts[np.arange(len(counts)), imax] += n - counts.sum(-1)
    assert (counts.sum(-1) == n).all(), np.unique(counts.sum(-1) - n, return_counts=True)
    return counts
    

def plot_matrix(matrix, xlabels=None, ylabels=None, **kw):
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest', **kw)
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90)
    if ylabels is not None:
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
    ax.grid(color='k', linestyle='-', linewidth=2)
    fig.colorbar(cax)
    

def main(annotations_file, video, features_file=None, win_size=30, plot=False):
    df = get_action_df(annotations_file, video)
    actions = df.narration.unique()
    print(actions)
    trans = get_transition_matrix(df, actions, win_size=win_size)
    emis = get_emission_matrix(actions)[:,1:]
    means, cov = get_gaussian_features(actions)
    means, cov = means[:,1:], cov[:,1:]

    if features_file:
        import h5py
        import torch
        from ptgprocess.egovlp import EgoVLP
        from hmmlearn.hmm import GaussianHMM, MultinomialHMM

        with h5py.File(features_file) as hf:
            print(set(hf))
            Z_video = hf['egovlp-n=10-fps=10'][:]
            i_frame = Z_video['frame']
            Z_video = Z_video['Z']
        model = EgoVLP()
        Z_text = model.encode_text(list(actions[1:])).cpu()
        sim = model.similarity(Z_text, torch.Tensor(Z_video)).numpy()
        plot_matrix(sim, None, actions)
        plt.savefig('sim-raw.png')

        hmmg = GaussianHMM(n_components=len(actions), covariance_type="diag")
        startprob = np.zeros(len(actions))
        startprob[0] = 1
        hmmg.startprob_ = startprob
        hmmg.transmat_ = trans
        hmmg.means_ = means
        hmmg.covars_ = cov

        actions_gauss = hmmg.predict(sim)
        pd.DataFrame({
            'frame': i_frame, 
            'top1': actions[1:][np.argmax(sim, axis=-1)], 
            'gmm': actions[actions_gauss]
        }).to_csv('gaus.csv', sep='\t')
        #print(list(actions[actions_gauss]))
        #plot_matrix(sim_gauss, None, actions)
        #plt.savefig('sim_gaussian.png')

        hmmn = MultinomialHMM(n_components=len(actions), n_trials=100)
        startprob = np.zeros(len(actions))
        startprob[0] = 1
        hmmn.startprob_ = startprob
        hmmn.transmat_ = trans
        hmmn.emissionprob_ = emis

        actions_mn = hmmn.predict(probs_to_obs_counts(sim, hmmn.n_trials))
        pd.DataFrame({
            'frame': i_frame, 
            'top1': actions[1:][np.argmax(sim, axis=-1)], 
            'gmm': actions[actions_mn]
        }).to_csv('mn.csv', sep='\t')
        #print(list(actions[actions_mn]))
        #sim_mn = np.zeros_like(sim)
        #sim_mn[:,actions_mn] = 1
        #plot_matrix(sim_mn, None, actions)
        #plt.savefig('sim_multinomial_100.png')
    
    np.savez('hmm.npz', trans=trans, emmission=emis, means=means, cov=cov)
    if plot:
        plot_matrix(trans, actions, actions)
        plt.savefig('transition.png')
        plot_matrix(emis, actions, actions)
        plt.savefig('emmission.png')
        plot_matrix(means, actions, actions)
        plt.savefig('mean.png')
        plot_matrix(cov, actions, actions)
        plt.savefig('cov.png')
    return trans

if __name__ == "__main__":
    import fire
    fire.Fire(main)
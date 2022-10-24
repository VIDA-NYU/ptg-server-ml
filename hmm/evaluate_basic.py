import os
import numpy as np
import pandas as pd
import tqdm
tqdm.tqdm.pandas()

import h5py

NO_ACTION = 'no action'

def get_action_df(df):
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
            'narration': NO_ACTION,
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
    return df

def load_Z_video(data_dir, video_id, key):
    fname = os.path.join(data_dir, video_id.split('_')[0], video_id + '.h5')
    with h5py.File(fname, 'r') as hf:
        tqdm.tqdm.write(str(set(hf)))
        Z_video = hf[key][:]
        i_frame = Z_video['frame']
        Z_video = Z_video['Z']
        return Z_video, i_frame

def eval_video(df, model, data_dir, video_id, all_actions, Z_text_all, topks=[1, 5, 10], offset=8):
    import torch
    df = get_action_df(df)
    narrations = df.narration.unique()
    actions = narrations[narrations != NO_ACTION]
    Z_text = model.encode_text(list(actions)).cpu()

    results = {}

    for key in ['egovlp-n=16-fps=16', 'egovlp-n=16-fps=8', 'egovlp-n=16-fps=4']:
        # get embeddings and output predictions
        z_video, frames = load_Z_video(data_dir, video_id, key)
        z_video = torch.Tensor(z_video)
    
        for vocab_name, (vocab, z_t) in {'video': (actions, Z_text), 'all': (all_actions, Z_text_all)}.items():
            y = model.similarity(z_t, z_video).numpy()

            # get ground truth
            gt = [
                set(df[(i >= df.start_frame) & (i < df.stop_frame)].narration)
                for i in frames - offset
            ]
            ranking = np.argsort(y, axis=-1)
            tqdm.tqdm.write(str(ranking.shape))

            resdf = pd.DataFrame({
                **{
                    f'top{k}': [bool(len(set(r) & g)) for r, g in zip(vocab[ranking[:, -k:]], gt)]
                    for k in topks
                },
                'frame': frames,
                'noaction': [g == {NO_ACTION} for g in gt]
            }).set_index(['noaction', 'frame'])


            results[(vocab_name, key.split('-', 1)[1])] = resdf
    return pd.concat(list(results.values()), keys=list(results), names=['vocab', 'window'])


def evaluate(annotations_file=None, data_dir='/vast/bs3639/EPIC-KITCHENS/egovlp', col='narration', suffix=''):
    if not annotations_file:
        final()
        return
    # load dataframe
    df = pd.read_csv(annotations_file)
    df['narration'] = df[col]
    print(col, len(df['narration'].unique()), 'unique actions')

    all_actions = df.narration.unique()
    all_actions = all_actions[all_actions != NO_ACTION]

    from ptgprocess.egovlp import EgoVLP
    model = EgoVLP()
    Z_text_all = model.encode_text(list(all_actions)).cpu()

    # do evaluation for each video
    results = {}
    for vid, df_vid in tqdm.tqdm(df.groupby('video_id')):
        results[vid] = eval_video(df_vid, model, data_dir, vid, all_actions, Z_text_all)
    result_df = pd.concat(list(results.values()), keys=list(results), names=['video_id'])

    result_df.to_csv(f'accuracy_raw{suffix or ""}.csv')

    # get aggregations
    aggregate(result_df, suffix=suffix)

def aggregate(result_df, suffix='', drop_noaction=True):
    if drop_noaction:
        print(result_df.index.get_level_values(-2).unique())
        print(result_df.index.get_level_values(-2).to_series().value_counts())
        result_df = result_df[result_df.index.get_level_values('noaction')==False]
    agg_df = result_df.groupby(level=[0,1,2]).mean().reset_index()
    total_df = result_df.groupby(level=[1,2]).mean(0).reset_index()
    agg_df = pd.concat([agg_df, total_df])
    agg_df.to_csv(f'accuracy_per_video{suffix or ""}.csv', index=False)
    print(agg_df.to_string())

def final(suffix=''):
    result_df = pd.read_csv(f'accuracy_raw{suffix or ""}.csv').set_index(['video_id','vocab','window', 'noaction', 'frame'])
    #result_df = result_df.iloc[:, 1:] # drop first column - weird artifact
    print(result_df.shape)
    aggregate(result_df, suffix=suffix)

if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)

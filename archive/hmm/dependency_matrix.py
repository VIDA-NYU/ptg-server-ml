import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_from_timed_actions_in_video(annotations_file, video, include_no_action=True, win_size=30, plot=False):

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
    ov_list = [i<=j for i,j in zip(df['start_frame'].tolist()[1:],df['stop_frame'].tolist()[:-1])]
    ov_list.insert(0, False)
    df['overlaps_previous'] = ov_list

    # create no action segments
    stop_frames = [r['start_frame'] for i, r in df.iterrows() if r['overlaps_previous']==False]
    start_frames = [df.iloc[i-1]['stop_frame'] for i, r in df.iterrows() if r['overlaps_previous']==False and i>0]
    indices = [i-0.5 for i,r in df.iterrows() if r['overlaps_previous']==False]
    if len(start_frames)<len(stop_frames):
        start_frames.insert(0,0) 
    noac_df = pd.DataFrame({
            'indices': indices,
            'narration': ['no action']*len(indices),
            'start_frame': start_frames,
            'stop_frame': stop_frames,
    })
    noac_df.set_index(['indices'], inplace=True)
    df = pd.concat([df,noac_df], axis=0, ignore_index=False).sort_index().drop(columns=['overlaps_previous']).reset_index().drop(columns=['index'])

    # check that all values are in good shape
    assert all(np.diff(df['start_frame'].tolist())>=0)
    assert all((df['stop_frame'].to_numpy() - df['start_frame'].to_numpy())>0)

    # create a column with each action duration
    df['nframes'] = df['stop_frame'].to_numpy() - df['start_frame'].to_numpy()

    ############################
    # create dependency matrix #
    ############################

    # get the unique actions
    actions2i = {v:i for i,v in enumerate(df['narration'].unique())}

    # initialize empty matrix to store values in
    matrix = np.zeros((len(actions2i),len(actions2i)))

    for i,r in df.iterrows():
        matrix[actions2i[r['narration']],actions2i[r['narration']]] += r['nframes']/win_size
        if i < len(df)-1:
            nr = df.loc[i+1]
            matrix[actions2i[nr['narration']],actions2i[r['narration']]] += 1

    matrix /= np.max(matrix,axis=1,keepdims=True)
    matrix = np.exp(matrix)/np.sum(np.exp(matrix),axis=1,keepdims=True)

    if plot:
        fig = plt.figure(figsize=(30,30))
        ax = fig.add_subplot(111)
        #cax = ax.matshow(np.exp(matrix/np.sum(matrix,axis=1)), interpolation='nearest')
        cax = ax.matshow(matrix, interpolation='nearest')
   
        alpha = list(actions2i.keys())
        xaxis = np.arange(len(alpha))
        ax.set_xticks(xaxis)
        ax.set_yticks(xaxis)
        ax.set_xticklabels(alpha, rotation=90)
        ax.set_yticklabels(alpha)
        ax.grid(color='k', linestyle='-', linewidth=2)
        
        plt.savefig('matrix.png')
    return matrix


if __name__ == "__main__":
    import fire
    fire.Fire(get_from_timed_actions_in_video)

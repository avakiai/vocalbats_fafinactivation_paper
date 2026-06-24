"""
npv0_wrangle.py — standardise raw annot_dat column names and values
"""

RENAME_COLS = {
    'animal':        'anim',
    'contrast':      'treatment',
    'n':             'rec_iter',
    'class_orig':    'call_type_manual',
    'class':         'call_type',
    't_last_onset':  'ici',
}

RENAME_SUFFIX_R = ['rms', 'spec_peak', 'spec_cent']

CONDITION_MAP  = {'silence':  'spontaneous'}
TREATMENT_MAP  = {'pre':      'saline', ### SHOULD BE CONTROL
                  'post':     'muscimol'}

COL_ORDER = ['anim', 'treatment', 'condition', 'rec_iter', 'start_seconds','stop_seconds','duration', 'ici', 'rec_dur',
             'rms_R', 'spec_peak_R', 'spec_cent_R', 'call_type', 'call_type_manual']


def wrangle_v0_cache_call_data(df):
    """
    Return a standardised copy of raw annot_dat.

    Steps
    -----
    1. Rename columns to project-standard names
    2. Append '_R' suffix to raw acoustic measure columns
    3. Replace condition / treatment values with standard labels
    4. Strip leading 'n' from rec_iter and cast to int
    5. Reorder columns
    """
    df = df.copy()
    df = df.rename(columns=RENAME_COLS)
    df = df.rename(columns={c: c + '_R' for c in RENAME_SUFFIX_R})
    df['condition']  = df['condition'].replace(CONDITION_MAP)
    df['treatment']  = df['treatment'].replace(TREATMENT_MAP)
    df['rec_iter']   = df['rec_iter'].str.lstrip('n').astype(int)
    df = df[COL_ORDER]
    return df


FILE_KEYS = ('anim', 'treatment', 'condition', 'rec_iter')


def get_file_keys(df, keys=FILE_KEYS):
    """
    Return a DataFrame of unique file-identifying combinations.

    Each row is one audio file's worth of calls. Iterate with:
        for _, file_id in get_file_keys(df).iterrows():
            file_df = df.loc[(df[keys] == file_id[list(keys)]).all(axis=1)]
    """
    return df[list(keys)].drop_duplicates().reset_index(drop=True)

from typing import Optional

import pandas as pd
import numpy as np

from source.utils.labels import ID_TO_CONDITION, ID_TO_LEVEL


class Submit:
    def __init__(
        self,
        base_submit_csv: Optional[str] = None,
        drop_row_ids: Optional[list[str]] = None,
    ):
        if base_submit_csv is not None:
            self.base_submit_df = pd.read_csv(base_submit_csv)
            self.base_submit_df['study_id'] = self.base_submit_df['row_id'].apply(lambda x: x.split('_')[0])
            self.base_submit_df['cond'] = self.base_submit_df['row_id'].apply(lambda x: x.split('_', 1)[1].rsplit('_', 2)[0])
        else:
            self.base_submit_df = None
        self.drop_row_ids = drop_row_ids

    def __call__(
        self,
        preds: np.ndarray,
        study_ids: np.ndarray,
        segment_ids: np.ndarray,
        condition_ids: np.ndarray,
        available_flags: Optional[np.ndarray] = None,
        labels: list[str] = ['row_id', 'normal_mild', 'moderate', 'severe'],
    ) -> pd.DataFrame:
        row_names = []

        if available_flags is None:
            available_flags = np.ones(len(study_ids), dtype=bool)

        available_mask = []
        for study_id, segment_id_list, condition_id_list, available_flag in zip(study_ids, segment_ids, condition_ids, available_flags):
            for condition_id in condition_id_list:
                for segment_id in segment_id_list:
                    condition = ID_TO_CONDITION[condition_id]
                    segment = ID_TO_LEVEL[segment_id]
                    if '/' in segment:
                        segment = segment.replace('/', '_').lower()
                    row_names.append(str(study_id) + '_' + condition + '_' + segment)
                    available_mask.append(available_flag)

        row_names = np.asarray(row_names)
        preds = preds.reshape(-1, len(labels[1:]))

        available_mask = np.asarray(available_mask)
        row_names = row_names[available_mask]
        preds = preds[available_mask]

        assert len(preds) == len(row_names)
        submit_df = pd.DataFrame()
        submit_df[labels[0]] = row_names
        submit_df[labels[1:]] = preds.reshape(-1, len(labels[1:]))

        if self.drop_row_ids is not None:
            # 指定した疾患の行を削除する
            # たとえば脊柱管狭窄を検出するモデルで補助ロスとしての他疾患を予測する場合、
            # 他疾患の結果は削除したい。よって本処理で削除する。
            submit_df = submit_df[~submit_df['row_id'].str.contains('|'.join(self.drop_row_ids))]

        if self.base_submit_df is not None:
            submit_df_condition_set = set([row_id.split('_', 1)[1].rsplit('_', 2)[0] for row_id in submit_df['row_id']])
            add_conditions = list(set(list(ID_TO_CONDITION.values())) - submit_df_condition_set)
            if len(add_conditions) > 0:
                study_id_set = set(study_ids.astype(str).tolist())
                add_df = self.base_submit_df[self.base_submit_df['study_id'].isin(study_id_set)].copy()
                add_df = add_df[add_df['cond'].isin(add_conditions)][['row_id', 'normal_mild', 'moderate', 'severe']]
                submit_df = pd.concat([submit_df, add_df], axis=0).reset_index(drop=True)

        return submit_df

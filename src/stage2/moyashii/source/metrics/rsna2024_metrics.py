# reference: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/521786

import copy

import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def get_condition(full_location: str) -> str:
    # Given an input like spinal_canal_stenosis_l1_l2 extracts 'spinal'
    for injury_condition in ['spinal', 'foraminal', 'subarticular']:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f'condition not found in {full_location}')


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    any_severe_scalar: float
) -> float:
    '''
    Pseudocode:
    1. Calculate the sample weighted log loss for each medical condition:
    2. Derive a new any_severe label.
    3. Calculate the sample weighted log loss for the new any_severe label.
    4. Return the average of all of the label group log losses as the final score, normalized for the number of columns in each group.
    This mitigates the impact of spinal stenosis having only half as many columns as the other two conditions.
    '''

    target_levels = ['normal_mild', 'moderate', 'severe']

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission[target_levels].values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission[target_levels].values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    solution['study_id'] = solution['row_id'].apply(lambda x: x.split('_')[0])
    solution['location'] = solution['row_id'].apply(lambda x: '_'.join(x.split('_')[1:]))
    solution['condition'] = solution['row_id'].apply(get_condition)

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)

    submission['study_id'] = solution['study_id']
    submission['location'] = solution['location']
    submission['condition'] = solution['condition']

    # すべての疾患が揃っていない状態でも動作するように処理を追加する
    target_conditions = submission['condition'].unique().tolist()

    condition_losses = []
    condition_weights = []
    for condition in target_conditions:
        condition_indices = solution.loc[solution['condition'] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, 'sample_weight'].values
        )
        condition_losses.append(condition_loss)
        condition_weights.append(1)

    if 'spinal' in target_conditions:
        # Derive a new any_severe label
        any_severe_spinal_labels = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['severe'].max())
        any_severe_spinal_weights = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['sample_weight'].max())
        any_severe_spinal_predictions = pd.Series(submission.loc[submission['condition'] == 'spinal'].groupby('study_id')['severe'].max())
        any_severe_spinal_loss = sklearn.metrics.log_loss(
            y_true=any_severe_spinal_labels,
            y_pred=any_severe_spinal_predictions,
            sample_weight=any_severe_spinal_weights
        )
        condition_losses.append(any_severe_spinal_loss)
        condition_weights.append(any_severe_scalar)

    return np.average(condition_losses, weights=condition_weights)


class RSNA2024Metrics:
    def __init__(
        self,
        train_df: pd.DataFrame,  # Pass train.csv as a DataFrame
        row_id_column_name="row_id",
        any_severe_scalar=1.0,
        sample_weights: dict[str, int] = {"normal_mild": 1, "moderate": 2, "severe": 4},
    ) -> None:
        self._train_df = train_df.copy()
        self._row_id_column_name = row_id_column_name
        self._any_severe_scalar = any_severe_scalar
        self._sample_weights = copy.deepcopy(sample_weights)

    def __call__(
        self,
        submission: pd.DataFrame,  # Pass submission.csv as a DataFrame
    ) -> float:
        """Convert train.csv to sample submission format with Nan values removed"""
        target_cols = list(self._sample_weights.keys())
        pred = submission.copy()  # Copy to prevent changes in original
        # Normalize values to have a sum of 1.0
        pred[target_cols] = pred[target_cols].div(pred[target_cols].sum(axis=1), axis=0)

        # Index the study_id in train_df
        train_df = self._train_df.copy()
        indexed_train_df = train_df.set_index("study_id", verify_integrity=True)

        row_ids = pred[self._row_id_column_name]
        study_ids = row_ids.apply(lambda x: x.split('_')[0])
        locations = row_ids.apply(lambda x: '_'.join(x.split('_')[1:]))

        solution_data = np.zeros_like(pred[target_cols].values)
        sample_weight_list = []
        nan_row_ids = set()
        for idx, (row, study_id, location) in enumerate(zip(row_ids, study_ids, locations)):
            severity = str(indexed_train_df.at[int(study_id), location]).replace("/", "_").lower()
            if severity in self._sample_weights:
                solution_data[idx, target_cols.index(severity)] = 1.0
                sample_weight_list.append(self._sample_weights[severity])
            else:
                solution_data[idx] = np.nan
                nan_row_ids.add(row)
                sample_weight_list.append(np.nan)

        solution = pd.DataFrame({
            self._row_id_column_name: pred[self._row_id_column_name],
            "sample_weight": sample_weight_list
        })
        solution[target_cols] = solution_data

        # Change row_ids in nan_row_ids to np.nan
        pred.loc[pred[self._row_id_column_name].isin(nan_row_ids), target_cols] = np.nan
        # Remove nan rows and pass copy to score function
        # score from https://www.kaggle.com/code/metric/rsna-lumbar-metric-71549?scriptVersionId=181722791 (Version 10)
        return score(solution.dropna().copy(), pred.dropna().copy(), self._row_id_column_name, self._any_severe_scalar)

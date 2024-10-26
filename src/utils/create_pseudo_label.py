import argparse
import pandas as pd
from src.utils import load_settings


def assign_pseudo_label_wrapper(label_columns: list[str], pseudo_df: pd.DataFrame) -> callable:
    mapping = {
        'Normal/Mild': [1.0, 0.0, 0.0],
        'Moderate': [0.0, 1.0, 0.0],
        'Severe': [0.0, 0.0, 1.0]
    }
    label_columns = label_columns
    pseudo_df = pseudo_df.copy()
    pseudo_df.set_index('row_id', inplace=True)

    def assign_pseudo_label(row):
        for col in label_columns:
            value = row[col]
            if pd.isnull(value):
                # NaNの場合は疑似ラベルを割り当てる
                row_id = f"{row['study_id']}_{col}"
                soft_label = pseudo_df.loc[row_id]
                row[col] = [soft_label['normal_mild'], soft_label['moderate'], soft_label['severe']]
            else:
                # NaNでない場合はワンホットラベルを割り当てる
                row[col] = mapping[value]
        return row

    return assign_pseudo_label


def main(args: argparse.Namespace):
    settings = load_settings()

    src_train_csv = settings.train_data_clean_dir / args.src_train_csv_name
    dst_train_csv = settings.train_data_clean_dir / args.dst_train_csv_name

    src_train_df = pd.read_csv(src_train_csv)
    spinal_df_list = []
    for csv_rel_path in args.spinal_csvs:
        csv_path = settings.model_checkpoint_dir / csv_rel_path
        df = pd.read_csv(csv_path)
        df = df[df['row_id'].str.contains('spinal')].reset_index(drop=True)
        if len(df) != (1975 * 1 * 5):
            raise ValueError(f"Invalid spinal csv: {csv_path}")
        spinal_df_list.append(df)

    lr_df_list = []
    for csv_rel_path in args.lr_csvs:
        csv_path = settings.model_checkpoint_dir / csv_rel_path
        df = pd.read_csv(csv_path)
        df = df[~df['row_id'].str.contains('spinal')].reset_index(drop=True)
        if len(df) != (1975 * 4 * 5):
            raise ValueError(f"Invalid lr csv: {csv_path}")
        lr_df_list.append(df)

    pred_df = pd.concat(spinal_df_list + lr_df_list, axis=0)
    pseudo_df = pred_df.groupby('row_id').mean().reset_index(drop=False)

    dst_train_df = src_train_df.copy()
    label_columns = dst_train_df.columns.drop(['study_id', 'fold']).tolist()
    dst_train_df = dst_train_df.apply(assign_pseudo_label_wrapper(label_columns, pseudo_df), axis=1)
    dst_train_df.to_csv(dst_train_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_train_csv_name', type=str)
    parser.add_argument('dst_train_csv_name', type=str)
    parser.add_argument('--spinal_csvs', type=str, nargs='+', required=True)
    parser.add_argument('--lr_csvs', type=str, nargs='+', required=True)
    args = parser.parse_args()
    main(args)

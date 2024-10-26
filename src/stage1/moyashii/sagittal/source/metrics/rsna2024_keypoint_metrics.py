# reference: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/521786


import numpy as np


class RSNA2024KeypointMetrics:
    def __init__(
        self,
        stride: int = 4,
        percentile: float = 99.8,
        score_name: str = 'acc(per)@0.02',
    ):
        self._stride = stride
        self._percentile = percentile
        self._score_name = score_name

    def _find_keypoint_by_max(self, preds: np.ndarray) -> np.ndarray:
        keypoints = []
        for heatmap_all_levels in preds:
            keypoints_per_image = []
            for heatmap in heatmap_all_levels:
                p_y, p_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
                p_y, p_x = (p_y + 0.5) * self._stride, (p_x + 0.5) * self._stride
                keypoints_per_image.append([p_x, p_y])
            keypoints.append(keypoints_per_image)
        return np.asarray(keypoints)

    def _find_keypoint_by_percentile(self, preds: np.ndarray, percentile: float) -> np.ndarray:
        keypoints = []
        for heatmap_all_levels in preds:
            keypoints_per_image = []
            for heatmap in heatmap_all_levels:
                threshold = np.percentile(heatmap, percentile)
                mask = heatmap >= threshold
                mask_indexes = np.where(mask)
                y_min = np.min(mask_indexes[0]) + 0.5
                y_max = np.max(mask_indexes[0]) + 0.5
                x_min = np.min(mask_indexes[1]) + 0.5
                x_max = np.max(mask_indexes[1]) + 0.5
                p_y, p_x = ((y_min + y_max) / 2 * self._stride, (x_min + x_max) / 2 * self._stride)
                keypoints_per_image.append([p_x, p_y])
            keypoints.append(keypoints_per_image)
        return np.asarray(keypoints)

    def _find_keypoint_by_centroid(self, preds: np.ndarray) -> np.ndarray:
        keypoints = []
        hmap_height, hmap_width = preds.shape[-2:]
        x_coords = np.arange(hmap_width)[np.newaxis, np.newaxis, :]
        y_coords = np.arange(hmap_height)[np.newaxis, :, np.newaxis]
        for heatmap_all_levels in preds:
            # ノイズ除去
            heatmap_all_levels[heatmap_all_levels < 0.1] = 0.0
            # 正規化
            heatmaps_sum = np.sum(heatmap_all_levels, axis=(1, 2), keepdims=True)
            norm_heatmap_all_levels = heatmap_all_levels / heatmaps_sum  # 正規化
            # x座標とy座標の重み付き平均（重心）を計算
            x_centers = np.sum(norm_heatmap_all_levels * x_coords, axis=(1, 2))
            y_centers = np.sum(norm_heatmap_all_levels * y_coords, axis=(1, 2))
            # キーポイント座標の出力
            keypoints_per_image = (np.vstack((x_centers, y_centers)).T + 0.5) * self._stride
            keypoints.append(keypoints_per_image)
        return np.asarray(keypoints)

    def _calc_normallized_distance(self, targets: np.ndarray, preds: np.ndarray, normalization_constant: float) -> np.ndarray:
        distances = np.linalg.norm(targets - preds, axis=2)
        return distances / normalization_constant

    def _calc_accuracy(self, norm_distance: np.ndarray, threshold: float) -> np.ndarray:
        # 画像ごとに正解かどうかを判定
        is_tp = np.all(norm_distance < threshold, axis=1)
        accuracy = np.mean(is_tp.astype(float))
        return accuracy

    def __call__(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
    ) -> float:
        # ヒートマップからキーポイントを取得
        max_keypoints = self._find_keypoint_by_max(preds)
        percentile_keypoints = self._find_keypoint_by_percentile(preds, self._percentile)
        centroid_keypoints = self._find_keypoint_by_centroid(preds)

        # 正規化ユークリッド距離を算出（画像の長辺を使って正規化）
        image_size = np.array(preds.shape[-2:]) * self._stride
        max_side_length = max(image_size)
        max_norm_dist = self._calc_normallized_distance(targets, max_keypoints, max_side_length)
        percentile_norm_dist = self._calc_normallized_distance(targets, percentile_keypoints, max_side_length)
        centroid_norm_dist = self._calc_normallized_distance(targets, centroid_keypoints, max_side_length)

        # メトリクスを計算（accuracyと平均正規化距離）
        metrics = {}
        for threshold in [0.01, 0.02, 0.03, 0.04, 0.05]:
            metrics[f'acc(max)@{threshold}'] = self._calc_accuracy(max_norm_dist, threshold)
            metrics[f'acc(per)@{threshold}'] = self._calc_accuracy(percentile_norm_dist, threshold)
            metrics[f'acc(cen)@{threshold}'] = self._calc_accuracy(centroid_norm_dist, threshold)
        metrics['mnd(max)'] = np.mean(max_norm_dist)
        metrics['mnd(per)'] = np.mean(percentile_norm_dist)
        metrics['mnd(cen)'] = np.mean(centroid_norm_dist)
        metrics['score'] = metrics[self._score_name]
        return metrics, max_norm_dist, percentile_norm_dist, centroid_norm_dist


if __name__ == '__main__':
    metrics = RSNA2024KeypointMetrics()
    targets = np.random.randint(0, 513, (10, 5, 2))
    preds = np.random.rand(10, 5, 128, 128)
    m = metrics(targets, preds)
    print(m)

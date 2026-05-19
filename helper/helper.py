def _extract_feature_statistics(pipeline) -> dict:
    """
    Ekstrak statistik per-feature dari StandardScaler di dalam pipeline.

    Return:
    {
        "feature_name": {
            "mean": float,
            "std": float,
            "normal_min": float,   # mean - 2*std (batas bawah 95%)
            "normal_max": float,   # mean + 2*std (batas atas 95%)
        },
        ...
    }
    """

    import numpy as np

    try:
        preprocessor = pipeline.named_steps.get("preprocessor")

        if preprocessor is None:
            return {}

        # Cari transformer numerik di ColumnTransformer
        for name, transformer, columns in preprocessor.transformers_:

            if not hasattr(transformer, "steps"):
                continue

            # Cari StandardScaler di dalam Pipeline transformer
            scaler = None
            for step_name, step in transformer.steps:
                if hasattr(step, "mean_") and hasattr(step, "scale_"):
                    scaler = step
                    break

            if scaler is None:
                continue

            # Pastikan jumlah feature cocok
            if len(columns) != len(scaler.mean_):
                continue

            feature_statistics = {}

            for i, col in enumerate(columns):

                mean = float(scaler.mean_[i])
                std = float(scaler.scale_[i])

                normal_min = mean - (2 * std)
                normal_max = mean + (2 * std)

                feature_statistics[col] = {
                    "mean": round(mean, 8),
                    "std": round(std, 8),
                    "normal_min": round(normal_min, 8),
                    "normal_max": round(normal_max, 8),
                }

            return feature_statistics

    except Exception as e:
        print(f"[feature_statistics] Gagal ekstrak: {e}")

    return {}

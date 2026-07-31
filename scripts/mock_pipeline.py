import numpy as np

from src.utils.parquet_writer import ParquetWriter


def run_mock_pipeline():
    output_path = "output/results.parquet"
    writer = ParquetWriter(output_path)

    frames = 300

    # Generate mock trajectories for Track ID 1 and Track ID 2
    track_ids = np.repeat([1, 2], frames).astype(np.int64)
    frame_indices = np.tile(np.arange(frames), 2).astype(np.int64)

    # Simulate movement across the screen
    x_coords = np.concatenate(
        [np.linspace(100, 400, frames), np.linspace(200, 500, frames)]
    ).astype(np.float32)

    y_coords = np.concatenate(
        [np.linspace(150, 300, frames), np.linspace(250, 400, frames)]
    ).astype(np.float32)

    z_coords = np.ones(frames * 2, dtype=np.float32) * 5.0
    ocr_texts = [""] * (frames * 2)
    confidences = np.ones(frames * 2, dtype=np.float32)

    writer.write_batch(
        track_ids=track_ids,
        frame_indices=frame_indices,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        ocr_texts=ocr_texts,
        confidences=confidences,
    )
    writer.close()
    print(f"Mock pipeline complete. Data written to {output_path}")


if __name__ == "__main__":
    run_mock_pipeline()

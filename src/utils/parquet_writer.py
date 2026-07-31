import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


# Define strict schema
TRACKING_SCHEMA = pa.schema(
    [
        ("track_id", pa.int64()),
        ("frame_idx", pa.int64()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("ocr_text", pa.string()),
        ("confidence", pa.float32()),
    ]
)


class ParquetWriter:
    """
    Writes tracking results to Parquet format with compression.
    """

    def __init__(
        self,
        output_path: str,
        compression: str = "snappy",  # snappy, gzip, brotli, lz4, zstd
        row_group_size: int = 10000,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.row_group_size = row_group_size
        self.writer = None

    def write_batch(
        self,
        track_ids: np.ndarray,
        frame_indices: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray,
        ocr_texts: List[str],
        confidences: np.ndarray,
    ) -> None:
        """
        Write a batch of tracking data to Parquet.

        Args:
            track_ids: Array of track IDs (int64)
            frame_indices: Array of frame indices (int64)
            x_coords: Array of x coordinates (float32)
            y_coords: Array of y coordinates (float32)
            z_coords: Array of depth values (float32)
            ocr_texts: List of OCR text strings
            confidences: Array of confidence scores (float32)
        """
        # Validate lengths
        n = len(track_ids)
        assert all(
            len(arr) == n
            for arr in [
                frame_indices,
                x_coords,
                y_coords,
                z_coords,
                ocr_texts,
                confidences,
            ]
        ), "All arrays must have same length"

        # Create PyArrow table
        table = pa.table(
            {
                "track_id": pa.array(track_ids, type=pa.int64()),
                "frame_idx": pa.array(frame_indices, type=pa.int64()),
                "x": pa.array(x_coords, type=pa.float32()),
                "y": pa.array(y_coords, type=pa.float32()),
                "z": pa.array(z_coords, type=pa.float32()),
                "ocr_text": pa.array(ocr_texts, type=pa.string()),
                "confidence": pa.array(confidences, type=pa.float32()),
            },
            schema=TRACKING_SCHEMA,
        )

        # Write to file
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.output_path,
                TRACKING_SCHEMA,
                compression=self.compression,
                use_dictionary=True,  # Compress repeated strings
                write_statistics=True,  # Enable query optimization
            )

        self.writer.write_table(table)
        print(f"[Parquet] Wrote {n} rows to {self.output_path}")

    def write_from_dict(self, data: Dict[str, np.ndarray]) -> None:
        self.write_batch(
            track_ids=data["track_id"],
            frame_indices=data["frame_idx"],
            x_coords=data["x"],
            y_coords=data["y"],
            z_coords=data["z"],
            ocr_texts=data["ocr_text"],
            confidences=data["confidence"],
        )

    def close(self) -> None:
        """Close writer and finalize file."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"[Parquet] Closed {self.output_path}")


def read_parquet(file_path: str) -> pd.DataFrame:
    return pd.read_parquet(file_path)

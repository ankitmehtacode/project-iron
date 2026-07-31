from typing import Dict
import numpy as np
import torch
from model_wrapper import ModelWrapper


class CoTracker3Wrapper(ModelWrapper):
    """Wrapper for CoTracker3 point tracking."""

    def load(self) -> None:
        from cotracker.predictor import CoTrackerPredictor

        self.model = CoTrackerPredictor(checkpoint=str(self.model_path))
        if torch.cuda.is_available() and self.device == "GPU":
            self.model = self.model.cuda()

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        video = torch.from_numpy(inputs["video"])
        if torch.cuda.is_available() and self.device == "GPU":
            video = video.cuda()

        # Grid tracking or query tracking
        if "grid_size" in inputs:
            tracks, visibility = self.model(
                video,
                grid_size=inputs["grid_size"],
                grid_query_frame=inputs.get("grid_query_frame", 0),
                backward_tracking=inputs.get("backward_tracking", False),
            )
        elif "queries" in inputs:
            queries = torch.from_numpy(inputs["queries"])
            if torch.cuda.is_available() and self.device == "GPU":
                queries = queries.cuda()
            tracks, visibility = self.model(
                video,
                queries=queries,
                backward_tracking=inputs.get("backward_tracking", False),
            )
        else:
            raise ValueError("Must provide either 'grid_size' or 'queries'")

        return {"tracks": tracks.cpu().numpy(), "visibility": visibility.cpu().numpy()}

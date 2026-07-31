import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

try:
    import openvino as ov
except ImportError:
    raise ImportError("OpenVINO not installed. Run: pip install openvino")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    def __init__(self, models_dir: str = "models/"):
        """Initialize the orchestrator."""
        self.models_dir = Path(models_dir)
        self.core = ov.Core()
        self.core.set_property({"ENABLE_MMAP": True})
        self.current_model = None
        self.current_request = None

        logger.info(
            f"Initialized InferenceOrchestrator with models_dir: {self.models_dir}"
        )

    def load_model(self, model_name: str, device: str = "CPU") -> None:
        """Load a single model using memory mapping."""
        if self.current_model is not None:
            self._cleanup()

        model_path = self.models_dir / f"{model_name}.xml"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model: {model_name} with mmap enabled")
        model = self.core.read_model(model=str(model_path))

        self.current_model = self.core.compile_model(model=model, device_name=device)
        self.current_request = self.current_model.create_infer_request()

        logger.info(f"Model {model_name} loaded successfully on {device}")

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.current_request is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Set inputs
        for name, data in inputs.items():
            self.current_request.set_input_tensor(data)

        # Run inference
        self.current_request.infer()

        # Get outputs
        outputs = {}
        for output in self.current_model.outputs:
            outputs[output.get_any_name()] = self.current_request.get_output_tensor(
                output.get_index()
            ).data

        return outputs

    def _cleanup(self) -> None:
        if self.current_request is not None:
            logger.info("Releasing inference request")
            self.current_request = None

        if self.current_model is not None:
            logger.info("Releasing compiled model")
            self.current_model = None

    def pipeline(
        self, model_sequence: list, initial_input: Dict[str, np.ndarray]
    ) -> Any:
        current_data = initial_input

        for model_name, transform_fn in model_sequence:
            logger.info(f"Running pipeline stage: {model_name}")

            # Load model
            self.load_model(model_name)

            # Run inference
            outputs = self.infer(current_data)

            # Transform output for next stage (if provided)
            if transform_fn is not None:
                current_data = transform_fn(outputs)
            else:
                current_data = outputs

            # Cleanup immediately after inference
            self._cleanup()

        return current_data


def main():
    orchestrator = InferenceOrchestrator(models_dir="models/")  # noqa: F841
    logger.info("Orchestrator ready. Add your models to models/ directory.")


if __name__ == "__main__":
    main()

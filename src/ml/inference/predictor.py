# ============================================================
# FILE: src/ml/inference/predictor.py
# ============================================================
"""
Model inference logic for clinical action recommendation.

Handles:
- Model loading
- Preprocessing
- Inference
- Action mapping
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.ml.models.resnet1d import resnet18_1d


class ClinicalActionPredictor:
    """Predicts clinical actions from ECG signals."""

    def __init__(
        self,
        model_path: Path | str,
        action_mapping_path: Path | str,
        device: str | None = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained PyTorch model (.pth file)
            action_mapping_path: Path to action mapping JSON
            device: Device to run inference on. Auto-detect if None.
        """
        self.model_path = Path(model_path)
        self.action_mapping_path = Path(action_mapping_path)

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.superclasses = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.model = self._load_model()

        # Load action mapping
        self.action_mapping = self._load_action_mapping()

    def _load_model(self) -> nn.Module:
        """Load trained model."""
        model = resnet18_1d(num_classes=len(self.superclasses), include_patient_context=False)

        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        model.eval()

        return model

    def _load_action_mapping(self) -> dict[str, list[dict[str, str]]]:
        """Load diagnostic → action mapping."""
        with open(self.action_mapping_path) as f:
            return json.load(f)

    def preprocess_signal(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess ECG signal.

        Args:
            signal: ECG signal, shape (12, num_samples) or (num_samples, 12)

        Returns:
            Preprocessed tensor, shape (1, 12, 5000)
        """
        # Ensure correct shape (12, num_samples)
        if signal.shape[0] != 12:
            signal = signal.T

        # Resample to 5000 samples if needed
        if signal.shape[1] != 5000:
            from scipy.signal import resample
            signal = resample(signal, 5000, axis=1)

        # Normalize per lead (z-score)
        signal = (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

        # Convert to tensor and add batch dimension
        signal_tensor = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)

        return signal_tensor

    def predict(
        self,
        signal: np.ndarray,
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Predict clinical actions from ECG signal.

        Args:
            signal: ECG signal, shape (12, num_samples) or (num_samples, 12)
            threshold: Probability threshold for diagnosis prediction
            top_k: Return top K recommendations

        Returns:
            Dictionary with:
                - diagnoses: List of predicted diagnoses with probabilities
                - recommendations: List of recommended clinical actions
        """
        # Preprocess
        signal_tensor = self.preprocess_signal(signal).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(signal_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Get predicted diagnoses
        diagnoses = []
        for i, (sc, prob) in enumerate(zip(self.superclasses, probs)):
            if prob > threshold:
                diagnoses.append({
                    'diagnosis': sc,
                    'confidence': float(prob),
                })

        # Sort by confidence
        diagnoses = sorted(diagnoses, key=lambda x: x['confidence'], reverse=True)

        # Map to actions
        recommendations = []
        seen_actions = set()  # Avoid duplicate actions

        for dx in diagnoses:
            dx_name = dx['diagnosis']
            if dx_name in self.action_mapping:
                for action_dict in self.action_mapping[dx_name]:
                    action_text = action_dict['action']
                    if action_text not in seen_actions:
                        recommendations.append({
                            'action': action_text,
                            'confidence': dx['confidence'],  # Inherit from diagnosis
                            'urgency': action_dict['urgency'],
                            'reasoning': action_dict['reasoning'],
                        })
                        seen_actions.add(action_text)

        # If no diagnoses above threshold, return normal
        if not recommendations:
            recommendations = self.action_mapping.get('NORM', [
                {
                    'action': 'Normal ECG - discharge with instructions',
                    'confidence': 1.0 - probs.max(),
                    'urgency': 'routine',
                    'reasoning': 'No significant abnormalities detected',
                }
            ])

        # Sort by urgency (immediate > urgent > routine), then by confidence
        urgency_order = {'immediate': 0, 'urgent': 1, 'routine': 2}
        recommendations = sorted(
            recommendations,
            key=lambda x: (urgency_order.get(x['urgency'], 3), -x['confidence'])
        )

        # Limit to top_k
        recommendations = recommendations[:top_k]

        return {
            'diagnoses': diagnoses,
            'recommendations': recommendations,
        }


# Singleton instance (lazy loading)
_predictor_instance = None


def get_predictor(
    model_path: Path | str | None = None,
    action_mapping_path: Path | str | None = None,
) -> ClinicalActionPredictor:
    """
    Get singleton predictor instance.

    Lazy loads model on first call. Subsequent calls return cached instance.
    """
    global _predictor_instance

    if _predictor_instance is None:
        # Default paths
        if model_path is None:
            model_path = Path(__file__).parents[3] / "models" / "best_model.pth"
        if action_mapping_path is None:
            action_mapping_path = Path(__file__).parent / "action_mapping.json"

        _predictor_instance = ClinicalActionPredictor(
            model_path=model_path,
            action_mapping_path=action_mapping_path,
        )

    return _predictor_instance

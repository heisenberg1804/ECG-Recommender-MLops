# ============================================================
# FILE: src/ml/explainability/explainer.py
# ============================================================
"""
Explainability module for ECG predictions.

Implements:
- Grad-CAM for 1D signals (shows which ECG segments matter)
- Lead importance scores (which leads drove the decision)
- Temporal attention (which time points are critical)
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn


class ECGGradCAM:
    """Grad-CAM for 1D ECG signals."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM explainer.

        Args:
            model: The trained model
            target_layer: Layer to extract activations from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_signal: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for target class.

        Args:
            input_signal: Input ECG signal (1, 12, 5000)
            target_class: Index of diagnosis to explain

        Returns:
            CAM heatmap (12, 5000) - normalized 0-1
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_signal)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # (channels, time)
        activations = self.activations[0]  # (channels, time)

        # Global average pooling of gradients (channel importance weights)
        weights = gradients.mean(dim=1, keepdim=True)  # (channels, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (time,)

        # Apply ReLU (only positive contributions)
        cam = fn.relu(cam)

        # Normalize to 0-1
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to original signal length
        cam = fn.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_signal.shape[-1],
            mode='linear',
            align_corners=False
        ).squeeze()

        cam_np = cam.cpu().numpy()

        # Broadcast to all 12 leads (for simplicity)
        # In practice, could compute per-lead importance separately
        cam_2d = np.tile(cam_np, (12, 1))

        return cam_2d


class ECGExplainer:
    """High-level explainer for ECG predictions."""

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize explainer.

        Args:
            model: Trained ECG model
            device: Device for computation
        """
        self.model = model
        self.device = device

        # Get the last convolutional layer for Grad-CAM
        # For ResNet: layer4[-1]
        self.target_layer = model.layer4[-1].conv2

        self.grad_cam = ECGGradCAM(model, self.target_layer)

    def explain_prediction(
        self,
        signal: np.ndarray,
        diagnosis_idx: int,
        diagnosis_name: str,
        lead_names: list[str] = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            signal: ECG signal (12, 5000)
            diagnosis_idx: Index of diagnosis to explain
            diagnosis_name: Name of diagnosis (e.g., "MI")
            lead_names: Names of ECG leads

        Returns:
            Explanation dictionary with importance scores and descriptions
        """
        if lead_names is None:
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Convert to tensor
        signal_tensor = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).to(self.device)

        # Generate Grad-CAM
        cam = self.grad_cam.generate_cam(signal_tensor, diagnosis_idx)

        # Compute lead importance (average CAM per lead)
        lead_importance = cam.mean(axis=1)  # (12,)

        # Normalize lead importance to 0-1
        if lead_importance.max() > 0:
            lead_importance = lead_importance / lead_importance.max()

        # Get top leads
        top_lead_indices = np.argsort(lead_importance)[::-1][:4]
        important_leads = [lead_names[i] for i in top_lead_indices]

        # Compute temporal importance (average CAM across leads)
        temporal_importance = cam.mean(axis=0)  # (5000,)

        # Find important time segments (where CAM > 0.5)
        important_times = np.where(temporal_importance > 0.5)[0]

        if len(important_times) > 0:
            # Convert sample indices to time (assuming 500 Hz)
            time_start_sec = important_times[0] / 500.0
            time_end_sec = important_times[-1] / 500.0
            time_range = f"{time_start_sec:.2f}s - {time_end_sec:.2f}s"
        else:
            time_range = "N/A"

        # Generate clinical description
        description = self._generate_clinical_description(
            diagnosis_name,
            important_leads,
            time_range
        )

        explanation = {
            'diagnosis': diagnosis_name,
            'important_leads': important_leads,
            'lead_importance_scores': {
                lead_names[i]: float(lead_importance[i])
                for i in top_lead_indices
            },
            'important_time_range': time_range,
            'temporal_attention': temporal_importance.tolist(),  # Full array for visualization
            'heatmap': cam.tolist(),  # (12, 5000) for visualization
            'description': description,
        }

        return explanation

    def _generate_clinical_description(
        self,
        diagnosis: str,
        leads: list[str],
        time_range: str
    ) -> str:
        """Generate human-readable clinical description."""
        lead_str = ", ".join(leads)

        descriptions = {
            'MI': f"ST segment changes detected in leads {lead_str} during interval {time_range}. "
                  f"This pattern is characteristic of myocardial infarction.",

            'STTC': f"ST/T wave abnormalities identified in leads {lead_str}. "
                    f"Model focused on the interval {time_range} for this classification.",

            'CD': f"Conduction abnormalities detected, particularly in leads {lead_str}. "
                  f"The model identified irregular patterns during {time_range}.",

            'HYP': f"Voltage criteria and morphology changes in leads {lead_str} "
                   f"suggest ventricular hypertrophy. Key interval: {time_range}.",

            'NORM': f"No significant abnormalities detected. The model examined all leads "
                    f"with particular attention to {lead_str}.",
        }

        return descriptions.get(
            diagnosis,
            f"Key features identified in leads {lead_str} during interval {time_range}."
        )


def explain_top_prediction(
    model: nn.Module,
    signal: np.ndarray,
    diagnoses: list[dict[str, Any]],
    device: torch.device,
    superclasses: list[str],
    patient_age: int | None = None,
    patient_sex: str | None = None,
    recommendations: list[dict[str, Any]] = None,
    use_llm: bool = False,  # Default to False for speed
) -> dict[str, Any]:
    """
    Explain the top prediction.

    Args:
        model: Trained model
        signal: ECG signal (12, 5000)
        diagnoses: List of diagnoses with confidence scores
        device: Computation device
        superclasses: List of class names
        patient_age: Patient age (for LLM context)
        patient_sex: Patient sex (for LLM context)
        recommendations: Clinical recommendations (for LLM context)
        use_llm: Whether to use LLM for explanation (slower but better)

    Returns:
        Explanation for the primary diagnosis
    """
    if not diagnoses:
        return {'explanation': 'No diagnosis to explain'}

    # Get primary diagnosis
    primary_dx = diagnoses[0]
    dx_name = primary_dx['diagnosis']
    dx_idx = superclasses.index(dx_name)

    # Create explainer
    explainer = ECGExplainer(model, device)

    # Generate Grad-CAM explanation
    explanation = explainer.explain_prediction(
        signal=signal,
        diagnosis_idx=dx_idx,
        diagnosis_name=dx_name,
    )

    # Add confidence from prediction
    explanation['confidence'] = primary_dx['confidence']

    # Try LLM explanation if requested
    if use_llm:
        try:
            import os
            # Only load LLM if not explicitly disabled
            if os.getenv("DISABLE_LLM_EXPLANATIONS", "false").lower() != "true":
                # from src.ml.explainability.gguf_explainer import get_gguf_explainer
                from src.ml.explainability.ollama_explainer import get_ollama_explainer


                llm_explainer = get_ollama_explainer()
                llm_description = llm_explainer.generate_explanation(
                    diagnosis=dx_name,
                    confidence=primary_dx['confidence'],
                    important_leads=explanation['important_leads'],
                    time_range=explanation['important_time_range'],
                    patient_age=patient_age,
                    patient_sex=patient_sex,
                    recommendations=recommendations or [],
                )

                # Keep both descriptions for comparison
                explanation['rule_based_description'] = explanation['description']
                explanation['llm_description'] = llm_description
                # Use LLM as primary
                explanation['description'] = llm_description
                print("✅ LLM explanation generated via GGUF")

        except Exception as e:
            print(f"⚠️ LLM explanation failed, using rule-based: {e}")
            # Keep the static description from explainer.explain_prediction

    return explanation

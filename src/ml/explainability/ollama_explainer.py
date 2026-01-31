# ============================================================
# FILE: src/ml/explainability/ollama_explainer.py
# ============================================================
"""
Ollama LLM-powered clinical explanations.

Uses local Ollama server for fast, private LLM inference.
"""
import os
from typing import Any

import requests


class OllamaClinicalExplainer:
    """Generate clinical explanations using Ollama."""

    def __init__(
        self,
        model_name: str = "hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama explainer.

        Args:
            model_name: Ollama model name (e.g., "llama3.2:3b", "biomistral", "mistral")
            base_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.base_url = base_url

        # Verify Ollama is running
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise ConnectionError("Ollama not reachable")
            print(f"✅ Connected to Ollama ({model_name})")
        except Exception as e:
            print(f"⚠️  Ollama not available: {e}")
            print("   Start with: ollama serve")
            raise

    def generate_explanation(
        self,
        diagnosis: str,
        confidence: float,
        important_leads: list[str],
        time_range: str,
        patient_age: int | None,
        patient_sex: str | None,
        recommendations: list[dict[str, Any]],
    ) -> str:
        """
        Generate clinical explanation using Ollama.

        Returns:
            2-3 sentence clinical explanation
        """
        # Build patient context
        patient_parts = []
        if patient_age:
            patient_parts.append(f"{patient_age}-year-old")
        if patient_sex:
            sex_full = "male" if patient_sex == "M" else "female"
            patient_parts.append(sex_full)
        patient_str = " ".join(patient_parts) if patient_parts else "patient"

        # Top actions
        top_actions = ", ".join([r['action'] for r in recommendations[:2]])

        # Build prompt
        prompt = f"""You are an expert cardiologist explaining ECG findings to a fellow physician.

ECG Analysis Results:
- Patient: {patient_str}
- Finding: {diagnosis}
- Model confidence: {confidence:.0%}
- Abnormal leads: {', '.join(important_leads)}
- Critical time interval: {time_range}
- Recommended: {top_actions}

Provide a concise 2-3 sentence clinical explanation that:
1. States what ECG changes were detected and where
2. Explains the clinical significance for this patient
3. Justifies the recommended actions

Use professional medical language. Be direct and clear.
Do not start with "The ECG shows" or "The model detected"
- write as if YOU found these abnormalities."""

        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower for consistency
                        "top_p": 0.9,
                        "num_predict": 150,  # Limit response length
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                explanation = result['response'].strip()

                # Clean up any preamble
                if explanation.startswith("Here"):
                    # Remove "Here is the explanation:" type prefixes
                    lines = explanation.split('\n')
                    explanation = '\n'.join(lines[1:]).strip()

                return explanation
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            print(f"⚠️ Ollama generation failed: {e}")
            return self._fallback_explanation(diagnosis, important_leads, time_range)

    def _fallback_explanation(
        self,
        diagnosis: str,
        leads: list[str],
        time_range: str
    ) -> str:
        """Fallback explanation if Ollama fails."""
        lead_str = ", ".join(leads)

        fallbacks = {
            'MI': f"ST segment elevation in leads {lead_str} during {time_range} indicates acute myocardial infarction requiring immediate intervention.",
            'STTC': f"ST/T wave abnormalities noted in leads {lead_str}. Urgent evaluation recommended to rule out ischemia.",
            'CD': f"Conduction disturbance detected in leads {lead_str}. Consider underlying structural abnormalities or medication effects.",
            'HYP': f"Increased voltages in leads {lead_str} meet criteria for ventricular hypertrophy. Echocardiography recommended.",
            'NORM': "Normal sinus rhythm with no acute abnormalities. All leads within normal limits.",
        }

        return fallbacks.get(diagnosis, f"Abnormalities detected in leads {lead_str} during interval {time_range}.")


# Singleton instance
_ollama_explainer = None


def get_ollama_explainer(
    model_name: str = "hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M"
) -> OllamaClinicalExplainer:
    """
    Get singleton Ollama explainer instance.

    Args:
        model_name: Ollama model to use
                   Options: "llama3.2:3b", "biomistral", "mistral:7b-instruct"
    """
    global _ollama_explainer

    # Check if disabled
    if os.getenv("DISABLE_LLM_EXPLANATIONS", "false").lower() == "true":
        raise ValueError("LLM explanations disabled")

    if _ollama_explainer is None:
        _ollama_explainer = OllamaClinicalExplainer(model_name=model_name)

    return _ollama_explainer

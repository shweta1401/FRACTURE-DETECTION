"""
GPT-Powered Diagnostic Summaries
- Generates radiology-style and layman explanations for fracture predictions
"""

import requests


def generate_gpt_summary(
    bone_type, fracture_status, confidence, cam_description,
    api_key, model="openai/gpt-3.5-turbo"
):
    """
    Generate GPT diagnostic summary.

    Args:
        bone_type (str): Predicted bone type
        fracture_status (str): "Fracture" or "No Fracture"
        confidence (float): Model-predicted fracture probability
        cam_description (str): Anatomical focus area from Grad-CAM++
        api_key (str): OpenRouter API key
        model (str): GPT model name (default: openai/gpt-3.5-turbo for lower cost)

    Returns:
        str: GPT-generated diagnostic summary
    """
    prompt = f"""
You are an AI medical assistant analyzing X-ray images for musculoskeletal fracture detection.

Bone type detected: {bone_type}
Fracture status: {fracture_status}
Model-predicted fracture probability: {confidence:.2f}
Grad-CAM analysis shows attention focused on: {cam_description}

Generate a concise radiology-style interpretation of this finding,
followed by a layman explanation for the patient.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 402:
            return (
                "⚠️ Unable to generate summary: Your OpenRouter account has no credits. "
                "Please top up your account or switch to a free model.\n\n"
                "👉 Tip: The default model is now `openai/gpt-3.5-turbo`, which may be more affordable. "
                "If you still face this issue, check your OpenRouter dashboard."
            )
        elif response.status_code == 401:
            return (
                "❌ Unauthorized: Your API key may be missing or invalid. "
                "Please add a valid OpenRouter API key in `.streamlit/secrets.toml`."
            )
        return f"❌ HTTP error occurred: {http_err}"
    except Exception as e:
        return f"❌ Unexpected error generating summary: {str(e)}"

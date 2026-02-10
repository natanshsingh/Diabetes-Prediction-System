import numpy as np

def generate_ai_explanation(input_data, feature_names, model):
    importances = model.feature_importances_
    contributions = input_data * importances

    top_indices = np.argsort(contributions)[-3:][::-1]

    explanation = "Based on the analysis, the AI observed:\n\n"

    for idx in top_indices:
        explanation += f"- **{feature_names[idx]}** has a strong influence on the prediction.\n"

    explanation += (
        "\nThe model compared your values with historical patient data "
        "and identified patterns commonly associated with diabetes risk."
    )

    return explanation
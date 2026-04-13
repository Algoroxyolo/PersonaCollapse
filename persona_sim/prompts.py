"""Prompt construction functions shared across all experiments."""

import json
from itertools import product


def create_persona_prompt(profile):
    """Convert a persona profile dictionary into a natural-language description.

    Args:
        profile: dict mapping dimension names to values.

    Returns:
        A string instructing the LLM to adopt the given persona.
    """
    persona_parts = []
    for dimension, value in profile.items():
        persona_parts.append(f"{dimension}: {value}")

    persona_description = "\n".join(persona_parts)
    return (
        "You are a person with the following background and values. "
        "When responding, speak entirely from this perspective-your thoughts, "
        "judgments, and reasoning should all reflect this identity:\n"
        + persona_description
    )


def create_scenario_prompt(scenario, persona_description):
    """Build a moral-dilemma evaluation prompt (1-5 Likert, A vs B).

    Args:
        scenario: The dilemma text narrated by an individual.
        persona_description: Output of :func:`create_persona_prompt`.

    Returns:
        The complete prompt string.
    """
    prompt = f"""{persona_description}
---
You will be given a moral dilemma described by an individual. You are not the person in the scenario-you are an impartial, third-person observer evaluating the situation from the outside.
Your task is to evaluate the ethical merits of two possible actions (A and B). Your evaluation should culminate in a decision expressed on a 5-point Likert scale, where:
    1 = Strongly favor A
    2 = Somewhat favor A
    3 = Neutral / undecided
    4 = Somewhat favor B
    5 = Strongly favor B

Respond with only a single number from 1 to 5.

---
Dilemma narrated by an individual (you are not the narrator!):
{scenario}
- A: The author is wrong.
- B: Others are wrong."""
    return prompt


def create_bfi_prompt(persona_description, question_text):
    """Build a BFI personality-test prompt (1-5 agreement scale).

    Args:
        persona_description: Output of :func:`create_persona_prompt`.
        question_text: The BFI item stem (e.g. "is talkative").

    Returns:
        The complete prompt string.
    """
    prompt = f"""{persona_description}

---
You are taking a personality test. Please indicate the extent to which you agree or disagree with the following statement describing you.
Statement: I see myself as someone who {question_text}

Response Scale:
1 = Disagree strongly
2 = Disagree a little
3 = Neither agree nor disagree
4 = Agree a little
5 = Agree strongly

Respond with ONLY a single number from 1 to 5."""
    return prompt


def generate_all_persona_profiles(dimensions):
    """Generate the Cartesian product of persona dimension values.

    Args:
        dimensions: dict mapping dimension name to list of possible values.

    Returns:
        A list of profile dicts.

    Raises:
        ValueError: If *dimensions* is empty or None.
    """
    if not dimensions:
        raise ValueError(
            "dimensions must be a non-empty dict. "
            "Use load_sampled_personas() for JSON-based persona loading."
        )

    profiles = []
    dimension_names = list(dimensions.keys())
    dimension_values = [dimensions[dim] for dim in dimension_names]

    for combination in product(*dimension_values):
        profile = {dim: val for dim, val in zip(dimension_names, combination)}
        profiles.append(profile)

    return profiles

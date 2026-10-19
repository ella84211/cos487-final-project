"""
This script uses Gemma to generate a semantic meaning for the formulas in the answers. 
It uses the LatexDataReader to read the answers from the data directory. 
The results are saved to a JSON file specified by the --output_file argument.
The script can be run with the --context flag to include the context of the answer in the prompt. 
It should be run as a module from the root of the repository. Assumes the /data/ directory contains the Posts.V1.3.xml file.

Example usage:
    python -m src.preprocessing.semantic_meanings --output_file data/semantic_meaning_results.json --context
"""

import json
from tqdm import tqdm
import os
import argparse
from pathlib import Path

from src.utils.reader import LatexDataReader
from src.preprocessing.gemma_model import load_model, generate, delete_model


# Number of characters to truncate the context to
TRUNCATE_LENGTH = 1000

MESSAGES_NO_CONTEXT = [
    {
        "role": "system",
        "content": (
            "You are a skilled mathematician. Given the list of LaTeX formulas, give the semantic meaning of the formulas. "
            "For example, if the formula is $x^2 + y^2 = z^2$, the semantic meaning is 'Pythagorean theorem'. "
            "The results will be used in bag-of-words retrieval, so the semantic meaning should be as specific and concise as possible. "
            "If the formulas are not well-known, give a brief description of what the formulas represent. "
            "Only provide the semantic meaning or description, in natural language (not in LaTeX). Do not enumerate the results. "
            "Do not include the original formula in your response. Your response should be less than 30 words. "
            "Do not leave any formulas delimited by $signs in your response. "
            "Do not provide any additional information, explanations, or context."
        )
    },
    {
        "role": "user",
        "content": ""
    },
]

MESSAGES_WITH_CONTEXT = [
    {
        "role": "system",
        "content": (
            "You are a skilled mathematician. Give the semantic meaning of the LaTeX formulas in the text. "
            "For example, if the text contains the formula $x^2 + y^2 = z^2$, the semantic meaning is 'Pythagorean theorem'. "
            "The results will be used in bag-of-words retrieval, so the semantic meaning should be as specific and concise as possible. "
            "If the formulas are not well-known, give a brief description of what the formulas represent. "
            "Only provide the semantic meaning or description, in natural language (not in LaTeX). Do not enumerate the results. "
            "Do not include the original formula in your response. Your response should be less than 30 words. "
            "Do not provide any additional information, explanations, or context."
        )
    },
    {
        "role": "user",
        "content": ""
    },
]


def get_formulas(text: str) -> list[str]:
    """Extracts LaTeX formulas from the given text.

    Args:
        text (str): The input text containing LaTeX formulas.
    Returns:
        list[str]: A list of extracted LaTeX formulas.
    """

    if text.count("$") % 2 != 0:
        return []

    formulas = []
    start = 0

    while True:
        start = text.find("$", start)
        if start == -1:
            break
        double = text.startswith("$$", start)
        delimiter = "$$" if double else "$"
        end = text.find(delimiter, start + len(delimiter))
        if end == -1:
            break
        formula = text[start + len(delimiter):end]
        formulas.append(formula)
        start = end + len(delimiter)
    return formulas


def main():
    parser = argparse.ArgumentParser(description="Process LaTeX formulas and extract semantic meanings.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file to save results.")
    parser.add_argument("--context", action="store_true", help="Use context for semantic meaning extraction.")
    args = parser.parse_args()

    output_file = args.output_file
    use_context = args.context

    filepath = Path(output_file)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    reader = LatexDataReader("data/", test=True)
    map_just_answers = reader.post_parser.map_just_answers
    model, assistant, processor = load_model()
    results = {}
    if use_context:
        messages = MESSAGES_WITH_CONTEXT
    else:
        messages = MESSAGES_NO_CONTEXT

    try:
        for id, answer in tqdm(map_just_answers.items()):
            answer_text = answer.body
            formulas = get_formulas(answer_text)
            if not formulas:
                continue
            results[id] = ""
            contents = []
            if use_context:
                len_answer_text = len(answer_text)
                i = 1
                while len_answer_text / i > TRUNCATE_LENGTH:
                    i += 1
                chunk_size = (len_answer_text + i - 1) // i
                for j in range(i):
                    chunk = answer_text[j * chunk_size:(j + 1) * chunk_size]
                    if chunk:
                        contents.append(f"Here is {'a chunk of' if i > 1 else 'the'} text: {chunk}")
            else:
                sum_len_formulas = sum(len(f) for f in formulas)
                if sum_len_formulas > TRUNCATE_LENGTH:
                    while sum_len_formulas > TRUNCATE_LENGTH and formulas:
                        formulas.pop()
                        sum_len_formulas = sum(len(f) for f in formulas)
                contents.append("Here are the formulas: " + ", ".join(formulas))
            for content in contents:
                if use_context:
                    messages[1]["content"] = content
                else:
                    messages[1]["content"] = content
                semantic_meaning = generate(
                    messages=messages,
                    model=model,
                    processor=processor,
                    assistant=assistant,
                    max_new_tokens=64
                )
                results[id] += semantic_meaning + " "
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"Results saved to {output_file}.")
    except (Exception, KeyboardInterrupt) as e:
        print(f"An error occurred: {e}")
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(f"Partial results saved to {output_file}.")


if __name__ == "__main__":
    main()

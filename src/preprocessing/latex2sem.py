"""
This script uses Gemma to generate semantic descriptions of LaTeX formulas
in cleaned answers and topic datasets.

Results are saved as JSON dictionaries mapping document IDs to semantic
descriptions. Separate output files are used for context and no-context modes.

By default, the script processes both answers and topics. Use --answers
to process only answers or --topics to process only topics.

The cleaned answers file defaults to data/clean/answers.jsonl.
The cleaned topics files default to the cleaned topic datasets for 2020–2022.

Use --context to include answer or topic text in the generation prompt.

Run the script as a module from the repository root.

Examples:
    python -m src.preprocessing.latex2sem                           # Process both answers and topics without context
    python -m src.preprocessing.latex2sem --answers                 # Process only answers without context
    python -m src.preprocessing.latex2sem --topics --context        # Process only topics with context
    python -m src.preprocessing.latex2sem --answers --context       # Process only answers with context

The with_context() and without_context() functions return dictionaries
mapping document IDs to cleaned text augmented with semantic descriptions.
"""

import json
from tqdm import tqdm
import argparse
from pathlib import Path

from pydetex.pipelines import strict_eqn

from src.preprocessing.gemma import load_model, generate
from src.utils import splice, get_formulas, get_text


# Number of characters to truncate the context to
TRUNCATE_LENGTH = 1000

data_dir = Path(__file__).resolve().parent.parent.parent / "data"
CLEANED_DATA = data_dir / "clean" / "answers.jsonl"
CLEANED_TOPICS_DATA = [
    data_dir / "clean" / "topics_2020.jsonl",
    data_dir / "clean" / "topics_2021.jsonl",
    data_dir / "clean" / "topics_2022.jsonl",
]
SEMANTIC_DESC_CONTEXT = data_dir / "semantic_descriptions" / "semantic_desc_with_context.json"
SEMANTIC_DESC_NO_CONTEXT = data_dir / "semantic_descriptions" / "semantic_desc_no_context.json"
SEMANTIC_DESC_CONTEXT_TOPICS = data_dir / "semantic_descriptions" / "semantic_desc_with_context_topics.json"
SEMANTIC_DESC_NO_CONTEXT_TOPICS = data_dir / "semantic_descriptions" / "semantic_desc_no_context_topics.json"

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


def with_context(source: Path = CLEANED_DATA, semantic_desc: Path = SEMANTIC_DESC_CONTEXT) -> dict:
    """Translate with Gemma, using context.

    Given a cleaned ``.jsonl`` file, returns ``{id: text}`` for every row in it.
    Assumes the semantic_desc file is a JSON file with {id: formula_desc} for every document
    that contains formulas.
    """
    if not _is_file(source):
        raise ValueError("with_context only supports .jsonl files.")
    return _add_descriptions(source, semantic_desc)



def without_context(source: Path = CLEANED_DATA, semantic_desc: Path = SEMANTIC_DESC_NO_CONTEXT) -> dict:
    """Translate with Gemma, without using context.

    Given a cleaned ``.jsonl`` file, returns ``{id: text}`` for every row in it.
    Assumes the semantic_desc file is a JSON file with {id: formula_desc} for every document
    that contains formulas.
    """
    if not _is_file(source):
        raise ValueError("without_context only supports .jsonl files.")
    return _add_descriptions(source, semantic_desc)


def with_context_topics(source: list[Path] = CLEANED_TOPICS_DATA, semantic_desc: Path = SEMANTIC_DESC_CONTEXT_TOPICS) -> dict:
    """Translate with Gemma, using context, for topics.

    Given a list of cleaned ``.jsonl`` files, returns one dictionary with ``{id: text}`` for every row in them.
    Assumes the semantic_desc file is a JSON file with {id: formula_desc} for every document
    that contains formulas.
    """
    results = {}
    for file in source:
        if not _is_file(file):
            raise ValueError("with_context_topics only supports .jsonl files.")
        results.update(_add_descriptions(file, semantic_desc))
    return results


def without_context_topics(source: list[Path] = CLEANED_TOPICS_DATA, semantic_desc: Path = SEMANTIC_DESC_NO_CONTEXT_TOPICS) -> dict:
    """Translate with Gemma, without using context, for topics.

    Given a list of cleaned ``.jsonl`` files, returns one dictionary with ``{id: text}`` for every row in them.
    Assumes the semantic_desc file is a JSON file with {id: formula_desc} for every document
    that contains formulas.
    """
    results = {}
    for file in source:
        if not _is_file(file):
            raise ValueError("without_context_topics only supports .jsonl files.")
        results.update(_add_descriptions(file, semantic_desc))
    return results


def _add_descriptions(source: Path, semantic_desc: Path) -> dict:
    if type(semantic_desc) is str:
        semantic_desc = Path(semantic_desc)
    if not semantic_desc.exists():
        raise FileNotFoundError(f"Semantic description file {semantic_desc} not found. See latex2sem documentation to generate it.")
    with open(semantic_desc, "r", encoding="utf-8") as file:
        data = json.load(file)
    cleaned_data = []
    with open(source, "r", encoding="utf-8") as file:
        for line in file:
            cleaned_data.append(json.loads(line))
    results = {}
    for item in cleaned_data:
        id = str(item["id"])
        text = get_text(item["body"])
        results[id] = " ".join(text).strip()
        if id in data:
            results[id] += " " + data[id]
    return results


def _is_file(source: str | Path) -> bool:
    return isinstance(source, Path) or source.endswith(".jsonl")


def generate_semantic_descriptions(data: list[dict], output_file: Path, model, assistant, processor, use_context: bool = False) -> None:
        results = {}
        if use_context:
            messages = MESSAGES_WITH_CONTEXT
        else:
            messages = MESSAGES_NO_CONTEXT

        try:
            for answer_obj in tqdm(data, desc="Processing answers"):
                formulas = get_formulas(answer_obj["body"])
                if not formulas:
                    continue
                id = str(answer_obj["id"])
                results[id] = ""
                contents = []
                if use_context:
                    answer_text = splice(answer_obj["body"])
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
                semantic_meaning = results[id].strip()
                if "\\" in semantic_meaning:
                    try:
                        cleaned_meaning = strict_eqn(
                            semantic_meaning,
                            show_progress=False
                        )
                        semantic_meaning = " ".join(cleaned_meaning.split())
                    except Exception:
                        pass
                results[id] = semantic_meaning.replace("$", "")

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)
            print(f"Results saved to {output_file}.")
        except (Exception, KeyboardInterrupt) as e:
            print(f"An error occurred: {e}")
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)
            print(f"Partial results saved to {output_file}.")


def main():
    parser = argparse.ArgumentParser(description="Process LaTeX formulas and extract semantic meanings.")
    parser.add_argument("--cleaned_file", type=str, default=CLEANED_DATA, help="Path to the cleaned JSONL file containing answers.")
    parser.add_argument("--cleaned_topics_files", nargs="*", default=CLEANED_TOPICS_DATA, help="Paths to the cleaned JSONL files containing topics.")
    parser.add_argument("--output_file_no_context", type=str, default=SEMANTIC_DESC_NO_CONTEXT, help="Path to the output JSON file to save results without context.")
    parser.add_argument("--output_file_context", type=str, default=SEMANTIC_DESC_CONTEXT, help="Path to the output JSON file to save results with context.")
    parser.add_argument("--output_topics_file_no_context", type=str, default=SEMANTIC_DESC_NO_CONTEXT_TOPICS, help="Path to the output JSON file to save results for topics without context.")
    parser.add_argument("--output_topics_file_context", type=str, default=SEMANTIC_DESC_CONTEXT_TOPICS, help="Path to the output JSON file to save results for topics with context.")
    parser.add_argument("--topics", action="store_true", help="Process topics, not answers.")
    parser.add_argument("--answers", action="store_true", help="Process answers, not topics.")
    parser.add_argument("--context", action="store_true", help="Use context for semantic meaning extraction.")
    args = parser.parse_args()

    model, assistant, processor = load_model()

    both = False
    if not args.topics and not args.answers:
        both = True

    if args.topics or both:
        cleaned_topics_files = args.cleaned_topics_files
        if args.context:
            output_topics_file = args.output_topics_file_context
        else:
            output_topics_file = args.output_topics_file_no_context
        use_context = args.context

        filepath = Path(output_topics_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for cleaned_file in cleaned_topics_files:
            with open(cleaned_file, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line))
        generate_semantic_descriptions(data, output_topics_file, model, assistant, processor, use_context)

    if args.answers or both:
        cleaned_file = args.cleaned_file
        if args.context:
            output_file = args.output_file_context
        else:
            output_file = args.output_file_no_context
        use_context = args.context

        filepath = Path(output_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = []
        with open(cleaned_file, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
        generate_semantic_descriptions(data, output_file, model, assistant, processor, use_context)


if __name__ == "__main__":
    main()

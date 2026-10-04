"""ARQMath readers that expose post and topic text with ``$...$`` formulas."""

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

_ARQ_ROOT = Path(__file__).resolve().parents[2] / "ARQMathCode"
if str(_ARQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARQ_ROOT))

from post_reader_record import DataReaderRecord
from topic_file_reader import TopicReader

_WHITESPACE = re.compile(r"\s+")
_BLOCK_TAGS = ("p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "pre", "tr")


def with_formula_delim(html: str | None) -> str:
    """Return visible text, with each math-container formula in ``$...$``.

    Collection posts store bare LaTeX inside ``<span class="math-container">``.
    Topic text often already uses ``$...$`` or ``$$...$$`` inside that span.
    A formula that already starts with ``$`` is left unchanged. Other HTML
    tags are removed. The XML files themselves are not modified.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for span in soup.find_all("span", class_="math-container"):
        formula = span.get_text().strip()
        if formula and not formula.startswith("$"):
            formula = f"${formula}$"
        span.replace_with(formula)

    for br in soup.find_all("br"):
        br.replace_with(" ")
    for block in soup.find_all(_BLOCK_TAGS):
        block.append(" ")

    text = soup.get_text(separator="")
    return _WHITESPACE.sub(" ", text).strip()


class LatexDataReader(DataReaderRecord):
    """``DataReaderRecord`` whose question and answer text uses ``$...$``.

    After the official reader loads the collection, question titles, question
    bodies, and answer bodies are replaced in memory. Pass the directory that
    holds ``Posts.V1.3.xml`` and the other collection files:

        reader = LatexDataReader("data", version=".V1.3")
        question = reader.post_parser.map_questions[1]
        question.body
    """

    def __init__(self, root_file_path: str, version: str = ".V1.3"):
        super().__init__(root_file_path, version)
        for question in self.post_parser.map_questions.values():
            question.title = with_formula_delim(question.title)
            question.body = with_formula_delim(question.body)
        for answer in self.post_parser.map_just_answers.values():
            answer.body = with_formula_delim(answer.body)


class LatexTopicReader(TopicReader):
    """``TopicReader`` whose title and question use ``$...$`` formulas.

        topics = LatexTopicReader("data/topics/Topics_Task1_2020.xml")
        topics.get_topic("A.1").question
    """

    def __init__(self, topic_file_path: str):
        super().__init__(topic_file_path)
        for topic in self.map_topics.values():
            topic.title = with_formula_delim(topic.title)
            topic.question = with_formula_delim(topic.question)

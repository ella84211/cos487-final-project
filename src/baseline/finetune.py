"""
Fine-tunes all-mpnet-base-v2 on ARQMath-1 and -2.
Writes to src/baseline/finetuned_model

Usage:
    python -m src.baseline.finetune
"""

import random
import html
import json
import shutil
import numpy as np
import torch

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from bs4 import BeautifulSoup
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ARQMathCode.Entity_Parser_Record.post_parser_record import PostParserRecord


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data"
POSTS = DATA_DIR / "Posts.V1.3.xml"

QREL_FILES = [
    DATA_DIR / "qrels" / "task1_2020.tsv",
    DATA_DIR / "qrels" / "task1_2021.tsv",
]

TOPIC_FILES = [
    DATA_DIR / "topics" / "Topics_Task1_2020.xml",
    DATA_DIR / "topics" / "Topics_Task1_2021.xml",
]

BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR = THIS_DIR / "finetuned_model"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512

VALIDATION_SPLIT = 0.10
K = 10
SEED = 42


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def parse_answer(text):
    soup = BeautifulSoup(text, "html.parser")
    for match in soup.find_all("span"):
        match.unwrap()
    return soup.get_text().strip()


def extract_plain_text(topic, tag_name):
    element = topic.find(tag_name)
    if element is None:
        return ""
    raw = element.get_text()
    decoded = html.unescape(raw)
    return BeautifulSoup(decoded, "html.parser").get_text().strip()


def get_topics():
    topics = {}
    for topic_file in TOPIC_FILES:
        year = topic_file.stem.split("_")[-1]
        print(f"Reading topics from {year}...")
        with open(topic_file, "r", encoding="utf-8") as f:
            xml = f.read()
        soup = BeautifulSoup(xml, "xml")
        for topic_xml in tqdm(soup.find_all("Topic"), desc=f"Topics {year}"):
            topic_id = topic_xml["number"]
            title = extract_plain_text(topic_xml, "Title")
            question = extract_plain_text(topic_xml, "Question")
            query = f"{title}\n{question}".strip()
            topics[(year, topic_id)] = query
    return topics


def get_qrels():
    qrels = defaultdict(dict)
    for qrel_file in QREL_FILES:
        year = qrel_file.stem.split("_")[-1]
        print(f"Reading qrels from {year}...")
        with open(qrel_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                topic_id, _, answer_id, relevance = (line.strip().split())
                relevance = float(relevance)
                if relevance > 0:
                    qrels[(year, topic_id)][str(answer_id)] = relevance
    return qrels


def split_queries(topics, qrels):
    available_queries = [key for key in qrels if key in topics]
    random.shuffle(available_queries)
    split_index = int(len(available_queries) * (1 - VALIDATION_SPLIT))
    train_keys = available_queries[:split_index]
    val_keys = available_queries[split_index:]

    train_qrels = {key: qrels[key] for key in train_keys}
    val_qrels = {key: qrels[key] for key in val_keys}
    train_topics = {key: topics[key] for key in train_keys}
    val_topics = {key: topics[key] for key in val_keys}
    print("\nDataset split:")
    print(f"Training queries: {len(train_keys)}")
    print(f"Validation queries: {len(val_keys)}")
    return train_topics, train_qrels, val_topics, val_qrels


def read_corpus(answer_ids):
    print("\nReading answer corpus...")
    post_reader = PostParserRecord(POSTS)
    corpus = {}
    for answer_id, post_obj in tqdm(post_reader.map_just_answers.items(), desc="Parsing answers"):
        answer_id = str(answer_id)
        if answer_id not in answer_ids:
            continue
        body = parse_answer(post_obj.body)
        if body:
            corpus[answer_id] = body
    print(f"Loaded {len(corpus)} answers.")
    return corpus


def create_examples(topics, qrels, corpus):
    examples = []
    missing = 0
    for topic_key, query in topics.items():
        relevant_answers = [answer_id for answer_id, relevance in qrels[topic_key].items() if relevance > 0 and answer_id in corpus]
        if not relevant_answers:
            missing += 1
            continue
        positive_id = random.choice(relevant_answers)
        positive_answer = corpus[positive_id]
        examples.append(InputExample(texts=[query, positive_answer]))
    print(f"Created {len(examples)} examples.")
    print(f"Queries without available answers: {missing}")
    return examples


def train_one_epoch(model, train_loss, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features = [{key: value.to(device) for key, value in feature.items()} for feature in features]
        labels = labels.to(device)
        optimizer.zero_grad()
        loss_value = train_loss(features, labels)
        loss_value.backward()
        optimizer.step()
        scheduler.step()

        batch_size = labels.shape[0]
        total_loss += loss_value.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def evaluate_loss(model, val_loss, dataloader):
    model.eval()
    device = model.device
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation loss", leave=False):
            features = [{key: value.to(device) for key, value in feature.items()} for feature in features]
            labels = labels.to(device)
            if labels.shape[0] < 2:
                continue
            loss_value = val_loss(features, labels)
            batch_size = labels.shape[0]
            total_loss += loss_value.item() * batch_size
            total_examples += batch_size
    if total_examples == 0:
        raise ValueError("No valid batches for validation loss.")
    return total_loss / total_examples


def dcg(relevances):
    relevances = np.asarray(relevances, dtype=float)
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    gains = (2 ** relevances - 1) / discounts
    return np.sum(gains)


def evaluate_ndcg(model, val_topics, val_qrels, corpus, k=10):
    model.eval()
    answer_ids = list(corpus.keys())
    answer_texts = list(corpus.values())
    corpus_embeddings = model.encode(
        answer_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    query_keys = list(val_topics.keys())
    query_texts = list(val_topics.values())
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    scores = np.matmul(query_embeddings, corpus_embeddings.T)
    ndcg_scores = []
    for i, topic_key in enumerate(query_keys):
        topic_qrels = val_qrels[topic_key]
        ranked_indices = np.argsort(scores[i])[::-1][:k]
        ranked_ids = [answer_ids[index] for index in ranked_indices]
        ranked_relevances = [topic_qrels.get(answer_id, 0.0) for answer_id in ranked_ids]
        actual_dcg = dcg(ranked_relevances)
        ideal_relevances = sorted(topic_qrels.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_relevances)
        if ideal_dcg == 0:
            continue
        ndcg_scores.append(actual_dcg / ideal_dcg)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def train():
    topics = get_topics()
    qrels = get_qrels()
    print(f"\nTotal topics: {len(topics)}")
    print(f"Queries with qrels: {len(qrels)}")

    train_topics, train_qrels, val_topics, val_qrels = split_queries(topics, qrels)
    answer_ids = set()
    for topic_qrels in qrels.values():
        answer_ids.update(topic_qrels.keys())
    corpus = read_corpus(answer_ids)
    train_examples = create_examples(
        train_topics,
        train_qrels,
        corpus
    )
    val_examples = create_examples(
        val_topics,
        val_qrels,
        corpus
    )
    if len(train_examples) < 2 or len(val_examples) < 2:
        raise ValueError("At least two training and validation examples are required.")

    print("\nLoading model...")
    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = min(MAX_SEQ_LENGTH, model.max_seq_length)
    print(f"Maximum sequence length: {model.max_seq_length}")
    device = model.device
    print(f"Device: {device}")

    train_batch_size = min(BATCH_SIZE, len(train_examples))
    val_batch_size = min(BATCH_SIZE, len(val_examples))
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=train_batch_size,
        drop_last=len(train_examples) >= train_batch_size,
        collate_fn=model.smart_batching_collate
    )
    val_dataloader = DataLoader(
        val_examples,
        shuffle=False,
        batch_size=val_batch_size,
        drop_last=False,
        collate_fn=model.smart_batching_collate
    )
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    val_loss = losses.MultipleNegativesRankingLoss(model=model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_ndcg = -1.0
    best_epoch = 0

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Validation metric: nDCG@{K}")

    history = []
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        training_loss = train_one_epoch(
            model,
            train_loss,
            train_dataloader,
            optimizer,
            scheduler,
            device
        )
        validation_loss = evaluate_loss(
            model,
            val_loss,
            val_dataloader
        )
        validation_ndcg = evaluate_ndcg(
            model,
            val_topics,
            val_qrels,
            corpus,
            k=K
        )
        print("\nEpoch results:")
        print(f"Training loss:       {training_loss:.4f}")
        print(f"Validation loss:     {validation_loss:.4f}")
        print(f"Validation nDCG@{K}:   {validation_ndcg:.4f}")

        history.append({
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_ndcg": validation_ndcg
        })
        if validation_ndcg > best_ndcg:
            best_ndcg = validation_ndcg
            best_epoch = epoch + 1
            model.save(str(OUTPUT_DIR))

            with open(OUTPUT_DIR / "training_info.json", "w") as f:
                json.dump({
                    "best_epoch": best_epoch,
                    "best_validation_ndcg": best_ndcg,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "validation_metric": f"nDCG@{K}",
                    "base_model": BASE_MODEL
                }, f, indent=4)
            print("\nNew best model saved!")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation nDCG@{K}: {best_ndcg:.4f}")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTraining history:")
    for item in history:
        print(
            f"Epoch {item['epoch']}: "
            f"Train loss={item['training_loss']:.4f}, "
            f"Val loss={item['validation_loss']:.4f}, "
            f"Val nDCG@{K}={item['validation_ndcg']:.4f}"
        )


if __name__ == "__main__":
    train()

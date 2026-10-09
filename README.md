# Preprocessing and Representation for Mathematical Information Retrieval

This repository is for a research project exploring how preprocessing and representation of mathematical formulas can improve information retrieval in the mathematical domain. This is the final project for COS487: Information Retrieval, at the University of Southern Maine.

### Introduction

In recent years, the capabilities of LLMs have increased significantly, as have neural information retrieval systems. Thus, researchers have applied these increasing capabilities to mathematical information retrieval to explore how we can better retrieve information within scientific domains. However, there remains an apparent lack of research exploring how offline preprocessing, including both traditional and LLM-based approaches, can enhance the capabilities of standard retrieval algorithms such as BM25 and TF/IDF. As such, we aim to explore whether preprocessing scientific information into more retrieval-friendly representations can reduce the need for online inference by performing more expensive processing offline, while returning to efficient standard retrieval algorithms online.

### Preprocessing Approaches

1. Translate LaTeX into natural language
   1. Using pydetex
   2. Using LLM translation
2. Extract abstract semantic language from LaTeX
   1. Using LaTeX only in LLM extraction
   2. Using LaTeX and its surrounding context in LLM extraction


## Setup

This project uses conda. `bin/install.sh` creates an environment named `cos487-final`, installs `requirements.txt`, and downloads the ARQMath Task 1 collection, topics, and qrels into `data/`.

```bash
bin/install.sh
conda activate cos487-final
```

The files come from the [ARQMath Drive folder](https://drive.google.com/drive/folders/1ZPKIWDnhMGRaPNVLi1reQxZWTfH2R4u3). Collection XML files go in `data/`. Topics and qrels go in `data/topics/` and `data/qrels/`.
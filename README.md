# Book Recommender System with Description Quality Control

## Overview
This project implements a book recommender system using semantic embeddings of book descriptions.

## Challenge
User-generated book descriptions are often noisy, incomplete, or low quality, which degrades embedding-based recommendations.

## Solution
We introduce an LLM-based quality control step that:
- filters low-quality descriptions
- fills missing descriptions
- improves semantic consistency before embedding
Missing book descriptions are enriched using a large language model. Generated descriptions are clearly marked and only used when no original description could be retrieved.

## Pipeline
1. Text cleaning
2. Description quality assessment (LLM)
3. Semantic embedding
4. Similarity-based recommendation

## Usage
```bash
python src/pipeline.py

LLM-generated descriptions were not quality-scored, as an initial audit showed consistently high quality across samples; they are used as a fallback when no sufficiently rated external description is available.
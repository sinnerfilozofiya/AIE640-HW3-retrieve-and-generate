# HW3: Retrieve and Generate

**Covers:** Weeks 7–8 (book chapters 7–8) · **Due:** Week 9, 26 Nov 2026, 23:59 (Turkey time) · **Weight:** 5% · **Expected effort:** 6–8 hours

Individual work. Runs on Google Colab with a free T4 GPU. Course policy: AI tools may be used for learning but not to complete the homework.

## The question

A team wants a question-answering assistant over their own documents. Before building it, they need to know which retriever finds the right documents, and whether retrieval and the choice of decoding actually make a small model's answers better.

## Part A: Which retriever? (Chapter 8) — 40 points

Data: [`BeIR/scifact`](https://huggingface.co/datasets/BeIR/scifact) — the `corpus` (5,183 abstracts; index `title + " " + text`) and the `queries`; relevance judgements from the `test` split of [`BeIR/scifact-qrels`](https://huggingface.co/datasets/BeIR/scifact-qrels). Evaluate on every query that has a judgement in that split.

1. **Lexical:** BM25 with `rank_bm25` (`BM25Okapi`, lower-cased whitespace tokens).
2. **Dense:** `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity.
3. **Hybrid:** reciprocal rank fusion of 1 and 2 (score = Σ 1 / (60 + rank)).
4. **Re-ranking:** re-rank the top 50 dense results with the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.

For each report Recall@10, nDCG@10 and MRR@10 (write the metric functions yourself and test them on a tiny example), and the time to answer all queries. Show 2 queries where BM25 and dense retrieval disagree most, with their top-3 titles.

## Part B: Does retrieval help the answer? (Chapters 7–8) — 45 points

Data: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) — `text-corpus` (passages) and the first 100 rows of `question-answer` (`test` split). Model: `Qwen/Qwen2.5-1.5B-Instruct` in half precision with the chat template; retrieve with `all-MiniLM-L6-v2`.

Compare three set-ups on the 100 questions (`max_new_tokens=64`):

1. **Closed-book**, greedy decoding: the question only.
2. **RAG**, greedy decoding: the top-3 passages in the prompt, with the instruction to answer only from them, briefly.
3. **RAG with beam search** (`num_beams=4`, `do_sample=False`), same prompt as 2.

For each report: **token F1** (SQuAD-style: lower-case, remove punctuation and articles, overlap of tokens with the gold answer), **answer containment** (share of predictions that contain the normalised gold answer), and seconds per question. Also report retrieval **hit@3** (share of questions whose gold answer appears in one of the 3 passages). Show 3 questions where RAG fixed the closed-book answer and 2 where it made it worse, and explain why.

## Part C: Recommendation — 15 points

Half a page: which retriever would you deploy, and why (quality vs latency vs no training data)? Is RAG worth its cost here, and is beam search worth its time? What should the team measure next on their own documents?

## Submission (one zip, fixed names)

Upload `hw3.zip` through the [submission portal](https://yunus.hacettepe.edu.tr/~hakanemekci/submit/) (course AIE640, assignment HW3). It must contain:

| File | Content |
|---|---|
| `hw3.ipynb` | Your notebook, **run top to bottom with all outputs visible** |
| `results.json` | Your main numbers in the format below |

```json
{
  "retrieval": {
    "bm25":   {"recall@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "dense":  {"recall@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "hybrid": {"recall@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0},
    "rerank": {"recall@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0}
  },
  "rag": {
    "hit@3": 0.0,
    "closed_book": {"token_f1": 0.0, "contains_answer": 0.0, "seconds_per_question": 0.0},
    "rag_greedy":  {"token_f1": 0.0, "contains_answer": 0.0, "seconds_per_question": 0.0},
    "rag_beam4":   {"token_f1": 0.0, "contains_answer": 0.0, "seconds_per_question": 0.0}
  },
  "recommended_retriever": "bm25 | dense | hybrid | rerank"
}
```

## Grading (100 points)

Points per part as above. Within each part: correctness 60%, code quality (readable, settings as specified, metric functions tested, no hard-coded results) 25%, analysis and insights 15%. The numbers in `results.json` must match the notebook outputs. Numbers are checked with tolerances that allow for differences between GPUs and library versions; timings are not compared with a reference. Late policy: 10% per day, up to 3 days; not accepted after that.

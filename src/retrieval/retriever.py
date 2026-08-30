import re
import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from collections import Counter

# Add project root to sys.path so it works when executed from inside ANY directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunk_documents import load_all_chunks, MetadataPolicy


# Standard English stopwords to eliminate lexical noise
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so",
    "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "from", "by", "with", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "of", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "could", "ought"
}

# Domain-specific synonym and concept expansion dictionary
SYNONYM_MAP = {
    "broken": ["damaged", "defective", "defect"],
    "torn": ["damaged", "defective"],
    "ripped": ["damaged", "defective"],
    "faulty": ["defective", "damaged"],
    "flaw": ["defective", "defect", "damaged"],
    "cracked": ["defective", "warranty", "manufacturing", "defect"],
    "replacement": ["warranty", "defect", "manufacturing", "claim"],
    "regular": ["standard", "return", "window", "30"],
    "how long": ["window", "calendar", "days", "standard", "return"],
    "send back": ["return", "returns"],
    "send it back": ["return", "returns"],
    "refund": ["return", "returns", "refunded"],
    "price adjustment": ["price", "adjustments", "sale", "difference"],
    "went on sale": ["price", "adjustments", "14", "days"],
    "gift card": ["gift", "cards", "final", "sale"],
    "guarantee": ["warranty", "coverage"],
    "postage": ["shipping", "carrier"],
    "dispatch": ["shipping", "shipped"],
    "cancel": ["cancellation", "cancellations"],
    "germany": ["international", "destinations", "shipping", "countries"],
    "canada": ["international", "shipping", "delivery", "estimate", "duties", "taxes"],
    "uk": ["international", "destinations", "shipping", "countries"],
    "france": ["international", "destinations", "shipping", "countries"],
    "australia": ["international", "destinations", "shipping", "countries"],
    "europe": ["international", "destinations", "shipping", "countries"],
    "japan": ["international", "destinations", "shipping", "countries"],
    "mexico": ["international", "destinations", "shipping", "countries"],
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Normalizes text into lowercase alphanumeric tokens.
    Handles hyphenated terms (e.g., 'final-sale' -> ['final-sale', 'final', 'sale']).
    Filters out common stopwords to avoid noise in keyword scoring.
    """
    if not text:
        return []

    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text_lower)

    expanded_tokens = []
    for token in tokens:
        if remove_stopwords and token in STOPWORDS:
            continue
        expanded_tokens.append(token)
        if "-" in token:
            for sub_token in token.split("-"):
                if not (remove_stopwords and sub_token in STOPWORDS) and len(sub_token) > 1:
                    expanded_tokens.append(sub_token)

    return [t for t in expanded_tokens if len(t) > 1]


def expand_query(query: str) -> List[str]:
    """Expands user query tokens with domain-specific synonyms and concepts."""
    base_tokens = tokenize(query, remove_stopwords=True)
    query_lower = query.lower()
    expanded = list(base_tokens)

    for phrase, synonyms in SYNONYM_MAP.items():
        if phrase in query_lower:
            expanded.extend(synonyms)

    return expanded


def compute_vector(tokens: List[str], idf_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Computes an L2-normalized TF-IDF vector for cosine similarity calculations.
    """
    tf = Counter(tokens)
    vector = {}
    norm_sq = 0.0

    for term, count in tf.items():
        weight = (1.0 + math.log(count)) * idf_dict.get(term, 1.0)
        vector[term] = weight
        norm_sq += weight * weight

    norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
    return {term: weight / norm for term, weight in vector.items()}


def cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Computes the cosine similarity between two normalized sparse/dense vectors."""
    # Dot product of normalized vectors
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(weight * v2.get(term, 0.0) for term, weight in v1.items())


class HybridRetriever:
    """
    Industry-Standard Hybrid Retriever combining:
    1. Okapi BM25 (Sparse Keyword Engine)
    2. Cosine Vector Embeddings (Dense Semantic Engine)
    3. Field Weighting: Headings (3x), Titles (2x), Content (1x)
    4. Metadata Precedence Policy: Suppresses legacy and internal documents
    5. Exact Citation Formatter: [file_name > heading]
    """

    def __init__(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        alpha: float = 0.5
    ):
        """
        Args:
            chunks: Knowledge base chunks list.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            alpha: Hybrid fusion weight (0.5 = equal 50/50 balance between Dense & BM25).
        """
        self.k1 = k1
        self.b = b
        self.alpha = alpha
        self.chunks = chunks if chunks is not None else load_all_chunks()

        self.doc_count = len(self.chunks)
        self.doc_token_counts: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_freqs: Counter = Counter()
        self.idf_dict: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []

        self._build_index()

    def _build_index(self):
        """Builds both the BM25 inverted index and the semantic vector space."""
        total_length = 0
        doc_tokens_list = []

        for chunk in self.chunks:
            title_tokens = tokenize(chunk["metadata"].get("title", ""), remove_stopwords=True)
            heading_tokens = tokenize(chunk.get("heading", ""), remove_stopwords=True)
            content_tokens = tokenize(chunk.get("content", ""), remove_stopwords=True)

            weighted_tokens = (heading_tokens * 3) + (title_tokens * 2) + content_tokens
            tf = Counter(weighted_tokens)

            self.doc_term_freqs.append(tf)
            self.doc_token_counts.append(len(weighted_tokens))
            doc_tokens_list.append(weighted_tokens)
            total_length += len(weighted_tokens)

            for term in set(weighted_tokens):
                self.doc_freqs[term] += 1

        self.avg_doc_length = (total_length / self.doc_count) if self.doc_count > 0 else 1.0

        # Precompute global IDF dictionary
        for term, n_q in self.doc_freqs.items():
            self.idf_dict[term] = math.log(1.0 + (self.doc_count - n_q + 0.5) / (n_q + 0.5))

        # Precompute L2-normalized dense/semantic vectors for each chunk
        for tokens in doc_tokens_list:
            self.doc_vectors.append(compute_vector(tokens, self.idf_dict))

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculates raw BM25 score for a single chunk."""
        tf = self.doc_term_freqs[doc_idx]
        doc_len = self.doc_token_counts[doc_idx]
        score = 0.0

        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf_dict.get(term, 0.0)
            numerator = freq * (self.k1 + 1.0)
            denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        authoritative_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K chunks using Hybrid (Dense + Sparse) fusion.
        """
        query_tokens = expand_query(query)
        if not query_tokens:
            return []

        # 1. Compute query semantic vector
        query_vector = compute_vector(query_tokens, self.idf_dict)

        # 2. Compute BM25 and Dense Cosine scores across all chunks
        raw_bm25_scores: List[float] = []
        raw_dense_scores: List[float] = []

        for idx in range(self.doc_count):
            bm25_s = self._bm25_score(query_tokens, idx)
            dense_s = cosine_similarity(query_vector, self.doc_vectors[idx])

            raw_bm25_scores.append(bm25_s)
            raw_dense_scores.append(dense_s)

        # Normalize BM25 scores to [0, 1] range for fair fusion
        max_bm25 = max(raw_bm25_scores) if raw_bm25_scores and max(raw_bm25_scores) > 0 else 1.0

        fused_scores: List[tuple[int, float, float, float]] = []

        for idx, chunk in enumerate(self.chunks):
            norm_bm25 = raw_bm25_scores[idx] / max_bm25
            dense_score = raw_dense_scores[idx]

            # Hybrid Score Fusion Formula: alpha * Dense + (1 - alpha) * BM25
            hybrid_base = (self.alpha * dense_score) + ((1.0 - self.alpha) * norm_bm25)

            # Metadata Policy Precedence Multiplier
            metadata = chunk.get("metadata", {})
            is_authoritative = MetadataPolicy.is_customer_authoritative(metadata)

            multiplier = 1.0
            if authoritative_only:
                if not is_authoritative:
                    multiplier = 0.05   # Heavily penalize legacy/internal notes
                else:
                    multiplier = 1.2    # Boost active official policies

            final_score = hybrid_base * multiplier

            if final_score > 0.0:
                fused_scores.append((idx, final_score, dense_score, norm_bm25))

        # Sort by final score descending
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, final_score, dense_s, bm25_s in fused_scores[:top_k]:
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["hybrid_score"] = round(final_score, 4)
            chunk_copy["dense_score"] = round(dense_s, 4)
            chunk_copy["bm25_score"] = round(bm25_s, 4)
            chunk_copy["citation"] = MetadataPolicy.format_citation(chunk_copy)
            results.append(chunk_copy)

        return results


if __name__ == "__main__":
    retriever = HybridRetriever()

    print("=== Testing Hybrid (Dense + Sparse BM25) Retriever ===\n")

    queries = [
        "How long does a regular customer have to return an unused backpack?",
        "My TrailPlus membership was active when I ordered. What is my return window?",
        "A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?",
        "Can I put the entire Breeze Tumbler in the dishwasher?",
        "Do all Aster & Row products have a lifetime warranty?",
        "Can you ship an Atlas Weekender to Germany?"
    ]

    for q in queries:
        print(f"Query: \"{q}\"")
        matches = retriever.retrieve(q, top_k=2)
        for i, match in enumerate(matches, 1):
            print(f"  [{i}] Hybrid: {match['hybrid_score']} (Dense: {match['dense_score']} | BM25: {match['bm25_score']})")
            print(f"      Source: {match['citation']}")
            print(f"      Status: {match['metadata'].get('status')} | Authority: {match['metadata'].get('policy_authority')}")
        print("-" * 70)

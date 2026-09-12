import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "TeachBack AI"
TAGLINE = "Turn passive notes into active understanding."

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Competition-style UI
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #fbfbfa;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .topline {
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 18px;
        margin-bottom: 22px;
    }

    .topline h1 {
        margin: 0;
        font-size: 2.1rem;
        color: #111827;
    }

    .topline p {
        margin: 6px 0 0 0;
        color: #6b7280;
        font-size: 1rem;
    }

    .concept-chip {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 5px 4px 0;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #dbe2ea;
        color: #334155;
        font-size: 0.86rem;
        font-weight: 600;
    }

    .result-box {
        border: 1px solid #dbe2ea;
        border-left: 4px solid #4f46e5;
        border-radius: 10px;
        padding: 18px;
        background: white;
        margin: 12px 0 16px 0;
    }

    .result-score {
        font-size: 2.2rem;
        line-height: 1;
        font-weight: 800;
        color: #111827;
    }

    .progress-strip {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr;
        gap: 12px;
        align-items: center;
        margin: 14px 0 4px 0;
    }

    .progress-cell {
        border: 1px solid #dbe2ea;
        border-radius: 10px;
        padding: 14px;
        background: white;
        text-align: center;
    }

    .progress-number {
        font-size: 1.55rem;
        font-weight: 800;
        color: #111827;
    }

    .progress-label {
        font-size: 0.78rem;
        color: #6b7280;
        margin-bottom: 4px;
    }

    .arrow {
        color: #64748b;
        font-size: 1.3rem;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.82rem;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 650;
        min-height: 44px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 10px;
    }

    button[data-baseweb="tab"] {
        border-radius: 8px;
        padding-left: 16px;
        padding-right: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data
# -----------------------------
STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","is","are","was","were","be","been",
    "being","as","at","by","from","that","this","these","those","it","its","into","than","then","can",
    "could","would","should","may","might","will","do","does","did","have","has","had","not","but","if",
    "they","their","them","we","our","you","your","i","he","she","his","her","which","what","when",
    "where","why","how","about","between","through","during","also","such","more","most","some","any"
}

SAMPLES = {
    "Machine Learning": """Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data rather than being explicitly programmed for every task. Supervised learning uses labelled examples to learn a mapping from inputs to outputs. Classification predicts discrete categories, while regression predicts continuous values. A model is trained on training data and evaluated on separate validation or test data so that performance reflects how well it generalises to unseen examples. Overfitting occurs when a model memorises training-specific noise and performs poorly on new data. Common ways to reduce overfitting include cross-validation, regularisation, simpler models, and collecting more representative data.""",

    "Biology": """Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll in chloroplasts absorbs light, while carbon dioxide enters leaves through stomata and water is absorbed by roots. During the light-dependent reactions, energy is used to form ATP and NADPH and oxygen is released. The Calvin cycle uses ATP and NADPH to help convert carbon dioxide into sugars. The rate of photosynthesis can be affected by light intensity, carbon dioxide concentration, temperature, and water availability.""",

    "Cybersecurity": """Phishing is a social engineering attack in which an attacker impersonates a trusted person or organisation to trick users into revealing sensitive information or taking unsafe actions. Common phishing messages create urgency and may contain malicious links or attachments. Multi-factor authentication can reduce the impact of stolen passwords. Users should verify suspicious requests through a separate trusted channel, inspect links carefully, and report suspected phishing attempts. Security awareness training helps users recognise common warning signs."""
}

CURATED_SAMPLE_CONCEPTS = {
    "Machine Learning": [
        "machine learning", "supervised learning", "classification", "regression",
        "training and test data", "generalisation", "overfitting", "regularisation"
    ],
    "Biology": [
        "photosynthesis", "chlorophyll", "stomata", "light-dependent reactions",
        "ATP and NADPH", "Calvin cycle", "carbon dioxide", "limiting factors"
    ],
    "Cybersecurity": [
        "phishing", "social engineering", "urgency", "malicious links",
        "multi-factor authentication", "trusted channel verification",
        "security awareness", "sensitive information"
    ],
}

CURATED_KEY_IDEAS = {
    "overfitting": [
        ("learns the training data too closely", ["training data too closely", "memorises training", "memorizes training", "memorisation", "memorization"]),
        ("training-specific noise or irrelevant patterns", ["training-specific noise", "training noise", "noise", "irrelevant patterns", "spurious patterns"]),
        ("poor generalisation to new or unseen data", ["poorly on unseen data", "poorly on new data", "unseen data", "new data", "unseen examples", "generalisation", "generalization"]),
        ("ways to reduce it, such as regularisation or cross-validation", ["regularisation", "regularization", "cross-validation", "simpler models", "representative data"]),
    ],
    "classification": [
        ("predicts discrete categories or classes", ["discrete categories", "categories", "classes", "class labels"]),
        ("maps an input to one of a set of labels", ["label", "labels", "one of", "category"]),
    ],
    "regression": [
        ("predicts continuous numerical values", ["continuous values", "continuous number", "numerical value", "numeric value"]),
        ("is used when the target is a quantity rather than a class", ["quantity", "number", "rather than a category", "not a class"]),
    ],
    "supervised learning": [
        ("learns from labelled examples", ["labelled examples", "labeled examples", "labelled data", "labeled data"]),
        ("learns a mapping from inputs to outputs", ["inputs to outputs", "input", "output", "mapping"]),
    ],
    "generalisation": [
        ("performance on new or unseen examples", ["unseen examples", "new examples", "unseen data", "new data"]),
        ("shows that the model learned transferable patterns rather than memorising", ["transferable", "not memorising", "not memorizing", "real patterns", "generalises"]),
    ],
    "training and test data": [
        ("training data is used to fit or learn the model", ["training data", "train the model", "fit the model"]),
        ("test or validation data is kept separate for evaluation", ["test data", "validation data", "kept separate", "evaluate"]),
    ],
    "regularisation": [
        ("reduces overfitting", ["reduce overfitting", "prevents overfitting", "limit overfitting"]),
        ("discourages overly complex fitting", ["complex", "penalty", "simpler", "constraints"]),
    ],
    "machine learning": [
        ("systems learn patterns from data", ["learn patterns", "patterns from data", "learn from data"]),
        ("rules do not have to be explicitly programmed for every task", ["not explicitly programmed", "without explicit programming", "learn rather than programmed"]),
    ],
    "photosynthesis": [
        ("converts light energy into chemical energy", ["light energy", "chemical energy"]),
        ("uses carbon dioxide and water", ["carbon dioxide", "water"]),
        ("produces sugars and releases oxygen", ["sugar", "glucose", "oxygen"]),
    ],
    "phishing": [
        ("an attacker impersonates a trusted person or organisation", ["impersonates", "pretends to be", "trusted person", "trusted organisation", "trusted organization"]),
        ("tries to trick users into revealing information or taking unsafe actions", ["trick users", "sensitive information", "unsafe actions", "steal information"]),
        ("often uses urgency, malicious links or attachments", ["urgency", "malicious links", "attachments", "suspicious link"]),
    ],
}

# -----------------------------
# Helpers
# -----------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


GENERIC_CONCEPT_WORDS = {
    "model", "models", "data", "training", "testing", "test", "learning",
    "machine", "algorithm", "algorithms", "system", "systems", "example",
    "examples", "method", "methods", "result", "results", "performance"
}

BAD_CONCEPT_FRAGMENTS = {
    "occurs model", "training-specific", "training specific",
    "overfitting occurs", "overfitting include", "models collecting"
}


def concept_is_meaningful(term: str) -> bool:
    t = clean_text(term).lower()
    if not t or t in BAD_CONCEPT_FRAGMENTS:
        return False
    if t.endswith(" occurs") or t.endswith(" include"):
        return False
    words = t.split()
    if len(words) == 1 and t in GENERIC_CONCEPT_WORDS:
        return False
    if len(words) == 1 and len(t) < 5:
        return False
    return True


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
    return [p.strip() for p in parts if len(p.strip()) > 25]


@st.cache_data(show_spinner=False)
def extract_concepts(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    sents = sentences(text)
    if not sents:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1200,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    )

    try:
        mat = vectorizer.fit_transform(sents)
    except ValueError:
        return []

    scores = np.asarray(mat.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = scores.argsort()[::-1]

    chosen = []
    seen_words = set()

    for idx in order:
        term = terms[idx]

        if not concept_is_meaningful(term):
            continue

        words = set(term.split())
        if words and words.issubset(seen_words):
            continue

        chosen.append((term, float(scores[idx])))
        seen_words |= words

        if len(chosen) >= top_n:
            break

    return chosen


@st.cache_data(show_spinner=False)
def relevant_context(text: str, concept: str, max_sentences: int = 3) -> str:
    sents = sentences(text)
    if not sents:
        return text

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    corpus = sents + [concept]

    try:
        mat = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(mat[:-1], mat[-1]).ravel()
        order = sims.argsort()[::-1][:max_sentences]
        picked = [sents[i] for i in order if sims[i] > 0]
        return " ".join(picked) if picked else " ".join(sents[:max_sentences])
    except ValueError:
        return " ".join(sents[:max_sentences])


@st.cache_resource(show_spinner=False)
def load_semantic_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _curated_ideas_for(concept: str):
    return CURATED_KEY_IDEAS.get(clean_text(concept).lower(), [])


def _idea_coverage(concept: str, answer: str):
    ideas = _curated_ideas_for(concept)
    if not ideas:
        return None, []
    a = _normalise_learning_language(answer)
    matched, missing = 0, []
    for label, variants in ideas:
        variants_norm = [_normalise_learning_language(v) for v in variants]
        if any(v in a for v in variants_norm):
            matched += 1
        else:
            missing.append(label)
    return matched / max(1, len(ideas)), missing


def _concept_from_reference(reference: str) -> str:
    low = clean_text(reference).lower()
    # Prefer the most distinctive curated concept whose name occurs in the reference.
    for concept in sorted(CURATED_KEY_IDEAS, key=len, reverse=True):
        if concept in low:
            return concept
    return ""


def fast_score(reference: str, answer: str, concept: str = "") -> Tuple[float, str]:
    """Fast, interpretable local score: lexical similarity + meaningful idea coverage."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        mat = vec.fit_transform([reference, answer])
        similarity = float(cosine_similarity(mat[0], mat[1])[0, 0])
    except ValueError:
        similarity = 0.0

    chosen = clean_text(concept).lower() or _concept_from_reference(reference)
    coverage, _ = _idea_coverage(chosen, answer)

    # Convert raw TF-IDF cosine into a friendlier 0..1 signal for short student answers.
    sim_norm = min(1.0, similarity / 0.42)
    if coverage is not None:
        score = 100.0 * (0.35 * sim_norm + 0.65 * coverage)
        # Correct concise answers should not be punished for not copying the source sentence.
        if coverage >= 0.50:
            score = max(score, 62.0 + 20.0 * (coverage - 0.50))
        if coverage >= 0.75:
            score = max(score, 80.0 + 12.0 * (coverage - 0.75) / 0.25)
    else:
        # Generic fallback for arbitrary pasted notes.
        ref_terms = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", reference.lower())) - STOPWORDS
        ans_terms = set(re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", answer.lower())) - STOPWORDS
        term_cov = len(ref_terms & ans_terms) / max(1, len(ref_terms))
        score = 100.0 * (0.72 * sim_norm + 0.28 * min(1.0, term_cov * 2.2))

    return max(0.0, min(100.0, score)), "Fast local NLP engine"

def deep_score(reference: str, answer: str) -> Tuple[float, str]:
    model = load_semantic_model()

    if model is None:
        return fast_score(reference, answer)

    emb = model.encode(
        [reference, answer],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    raw = float(np.dot(emb[0], emb[1]))
    score = max(0.0, min(100.0, (raw - 0.15) / 0.70 * 100.0))
    return score, "Sentence-BERT semantic similarity"


def semantic_score(reference: str, answer: str, mode: str, concept: str = "") -> Tuple[float, str]:
    if mode == "Fast":
        return fast_score(reference, answer, concept)
    deep, engine = deep_score(reference, answer)
    coverage, _ = _idea_coverage(clean_text(concept).lower(), answer)
    if coverage is not None:
        deep = max(0.0, min(100.0, 0.55 * deep + 45.0 * coverage))
    return deep, engine


def important_terms(text: str, top_n: int = 10) -> List[str]:
    return [x[0] for x in extract_concepts(text, top_n=top_n)]


def concept_match_scores(notes: str, concepts: List[str], answer: str):
    """Compare the learner's explanation with the note context for every extracted concept."""
    contexts = [relevant_context(notes, c) for c in concepts]
    corpus = contexts + [answer]

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        mat = vec.fit_transform(corpus)
        sims = cosine_similarity(mat[:-1], mat[-1]).ravel()
    except ValueError:
        sims = np.zeros(len(concepts))

    return {concept: float(score) for concept, score in zip(concepts, sims)}


def detect_concept_mismatch(
    notes: str,
    concepts: List[str],
    selected: str,
    answer: str,
):
    """
    Detect a likely wrong-concept explanation without treating generic words
    like 'model' or 'data' as alternative concepts.
    """
    answer_low = clean_text(answer).lower()
    selected_low = selected.lower().strip()

    alternatives = [
        c for c in concepts
        if c != selected and concept_is_meaningful(c)
    ]

    # Strongest signal: learner explicitly names a different meaningful concept.
    mentioned = []
    for concept in alternatives:
        c = concept.lower().strip()
        if re.search(rf"\b{re.escape(c)}\b", answer_low):
            mentioned.append(concept)

    selected_named = bool(
        re.search(rf"\b{re.escape(selected_low)}\b", answer_low)
    )

    if mentioned and not selected_named:
        detected = max(mentioned, key=lambda x: (len(x.split()), len(x)))
        return {
            "selected": selected,
            "detected": detected,
            "confidence": 98,
            "reason": "explicit_concept",
        }

    # Secondary signal: only flag similarity mismatch when evidence is strong.
    compare_concepts = [selected] + alternatives
    scores = concept_match_scores(notes, compare_concepts, answer)
    if not scores:
        return None

    selected_score = scores.get(selected, 0.0)
    ranked = sorted(
        [(c, s) for c, s in scores.items() if c != selected],
        key=lambda x: x[1],
        reverse=True,
    )
    if not ranked:
        return None

    best_concept, best_score = ranked[0]

    if (
        best_score >= 0.22
        and selected_score <= 0.12
        and (best_score - selected_score) >= 0.12
    ):
        return {
            "selected": selected,
            "detected": best_concept,
            "confidence": min(92, round(60 + (best_score - selected_score) * 120)),
            "reason": "context_similarity",
        }

    return None

def weak_reference(context: str, weak_idea: str) -> str:
    """Find the most relevant source sentence for a weak idea."""
    sents = sentences(context)
    if not sents:
        return context

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        mat = vec.fit_transform(sents + [weak_idea])
        sims = cosine_similarity(mat[:-1], mat[-1]).ravel()
        return sents[int(np.argmax(sims))]
    except ValueError:
        return context


def diagnostic_question(selected: str, misses: List[str], context: str) -> Tuple[str, str]:
    """
    Build one targeted diagnostic question from the learner's weakest idea.
    Returns (question, reference_text_for_scoring).
    """
    weak = misses[0] if misses else selected
    low = weak.lower()
    reference = weak_reference(context, weak)

    if "noise" in low:
        q = f"Why could learning {weak} make a model perform worse on new or unseen examples?"
    elif "general" in low:
        q = f"Why is {weak} important when deciding whether a model has really learned?"
    elif "training" in low and "data" in low:
        q = f"How should {weak} be used without confusing memorisation with real learning?"
    elif "test" in low or "validation" in low:
        q = f"Why should {weak} be kept separate when checking whether learning transfers to new examples?"
    elif "classification" in low:
        q = f"When would {weak} be more appropriate than predicting a continuous number?"
    elif "regression" in low:
        q = f"When would {weak} be more appropriate than predicting a category?"
    else:
        q = (
            f"You were less clear about '{weak}'. "
            f"Why is this idea important for understanding {selected}, and what would happen if it were ignored?"
        )

    return q, reference




def score_challenge_answer(reference: str, answer: str, weak_idea: str, mode: str = "Fast") -> Tuple[float, str]:
    """Score a focused diagnostic answer without expecting the learner to restate the whole concept."""
    # Lexical similarity gives a small signal, while targeted idea coverage matters more.
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        mat = vec.fit_transform([reference, answer])
        similarity = float(cosine_similarity(mat[0], mat[1])[0, 0])
    except ValueError:
        similarity = 0.0

    a = _normalise_learning_language(answer)
    weak = _normalise_learning_language(weak_idea)

    weak_terms = [
        w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{3,}\b", weak)
        if w not in STOPWORDS and w not in {"such", "ways", "idea", "important"}
    ]
    target_cov = (sum(term in a for term in weak_terms) / max(1, len(weak_terms))) if weak_terms else 0.0

    # Reward a clear cause/consequence connection in diagnostic answers.
    consequence_markers = [
        "new data", "unseen data", "new examples", "unseen examples",
        "performs worse", "perform worse", "poorly", "generalisation", "generalization"
    ]
    consequence = 1.0 if any(marker in a for marker in consequence_markers) else 0.0

    # Common equivalent wording for training-specific noise / irrelevant patterns.
    if any(x in weak for x in ["noise", "irrelevant patterns", "spurious patterns"]):
        idea_hit = 1.0 if any(x in a for x in ["noise", "irrelevant patterns", "spurious patterns", "random patterns"]) else target_cov
    else:
        idea_hit = target_cov

    sim_norm = min(1.0, similarity / 0.30)
    score = 100.0 * (0.25 * sim_norm + 0.50 * idea_hit + 0.25 * consequence)

    # A focused answer that clearly states the weak idea and consequence should read as strong.
    if idea_hit >= 0.75 and consequence >= 1.0:
        score = max(score, 82.0)
    elif idea_hit >= 0.50:
        score = max(score, 68.0)

    if mode == "Deep analysis":
        model = load_semantic_model()
        if model is not None:
            emb = model.encode([reference, answer], normalize_embeddings=True, show_progress_bar=False)
            semantic = max(0.0, min(100.0, (float(np.dot(emb[0], emb[1])) - 0.10) / 0.65 * 100.0))
            score = 0.65 * score + 0.35 * semantic
            return max(0.0, min(100.0, score)), "Focused diagnostic + Sentence-BERT"

    return max(0.0, min(100.0, score)), "Focused diagnostic scoring"

def challenge_feedback(score: float) -> str:
    if score >= 75:
        return "Strong answer — you connected the weak idea back to the concept."
    if score >= 50:
        return "Partly there — the connection is visible, but explain the cause or consequence more clearly."
    return "This gap is still active. Review the source idea, then answer the question in your own words."


def _normalise_learning_language(text: str) -> str:
    """
    Light synonym normalisation so equivalent student wording is not
    incorrectly marked as missing in Fast mode.
    """
    t = clean_text(text).lower()

    synonym_groups = {
        "new data": [
            "unseen data", "new data", "novel data", "unknown data",
            "new examples", "unseen examples", "future data"
        ],
        "generalisation": [
            "generalization", "generalisation", "work on new data",
            "perform on new data", "works on unseen data",
            "performs on unseen data"
        ],
        "training data": [
            "training set", "training dataset", "training examples"
        ],
        "test data": [
            "test set", "testing data", "testing set", "held out data",
            "held-out data"
        ],
        "training-specific noise": [
            "training noise", "noise in training data",
            "random noise", "irrelevant patterns", "spurious patterns"
        ],
    }

    for canonical, variants in synonym_groups.items():
        for variant in variants:
            if variant in t:
                t = t.replace(variant, canonical)

    return t


def missing_concepts(reference: str, answer: str, top_n: int = 7, concept: str = "") -> List[str]:
    chosen = clean_text(concept).lower() or _concept_from_reference(reference)
    coverage, curated_missing = _idea_coverage(chosen, answer)
    if coverage is not None:
        return curated_missing[:top_n]

    # Fallback for arbitrary notes: keep only cleaner noun-like TF-IDF phrases.
    ans = _normalise_learning_language(answer)
    misses = []
    blocked_single = {
        "occurs", "memorises", "memorizes", "predicts", "values", "performs",
        "uses", "learns", "include", "includes", "collecting", "evaluated",
        "trained", "reflects", "enables"
    }
    for term in important_terms(reference, top_n=18):
        normal_term = _normalise_learning_language(term)
        words = [w for w in normal_term.split() if w not in STOPWORDS]
        if not words or (len(words) == 1 and words[0] in blocked_single):
            continue
        if len(words) == 2 and words[0] in {"model", "data", "training"} and words[1] in blocked_single:
            continue
        covered = normal_term in ans or (sum(w in ans for w in words) / len(words) >= 0.75)
        if not covered and concept_is_meaningful(term):
            misses.append(term)
        if len(misses) >= top_n:
            break
    return misses

def clean_missing_concepts(misses: List[str], max_items: int = 5) -> List[str]:
    cleaned = []
    seen = set()

    for item in misses:
        value = clean_text(item)
        low = value.lower()

        if not concept_is_meaningful(value):
            continue

        if low in {"training-specific", "training specific"}:
            continue

        if low not in seen:
            cleaned.append(value)
            seen.add(low)

        if len(cleaned) >= max_items:
            break

    return cleaned

def feedback(score: float, misses: List[str]) -> Tuple[str, str]:
    if score >= 82:
        level = "Strong understanding"
        msg = "You explained the main idea clearly. Your next goal is precision."
    elif score >= 65:
        level = "Good, but incomplete"
        msg = "You understand the core idea, but a few important links are still missing."
    elif score >= 45:
        level = "Partial understanding"
        msg = "You recognise the topic, but your explanation needs more of the key ideas."
    else:
        level = "Needs another pass"
        msg = "Review the highlighted ideas, then explain the concept again in your own words."

    if misses:
        msg += " Focus on: " + ", ".join(misses[:4]) + "."

    return level, msg


def make_cloze_questions(text: str, concepts: List[str], n: int = 5):
    sents = sentences(text)
    questions = []
    used = set()

    for concept in concepts:
        for s in sents:
            m = re.search(rf"\b{re.escape(concept)}\b", s, flags=re.I)
            if m and s not in used:
                blanked = s[:m.start()] + "_____" + s[m.end():]
                questions.append((blanked, concept))
                used.add(s)
                break

        if len(questions) >= n:
            break

    if len(questions) < n:
        for s in sents:
            words = [
                w for w in re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", s)
                if w.lower() not in STOPWORDS
            ]
            if not words or s in used:
                continue

            target = max(words, key=len)
            blanked = re.sub(
                rf"\b{re.escape(target)}\b",
                "_____",
                s,
                count=1,
                flags=re.I
            )
            questions.append((blanked, target))
            used.add(s)

            if len(questions) >= n:
                break

    return questions


def mastery_band(score: float) -> str:
    if score >= 82:
        return "Mastered"
    if score >= 65:
        return "Developing"
    return "Review"


def latest_two_scores(history, concept):
    scores = [h["score"] for h in history if h["concept"] == concept]
    if len(scores) >= 2:
        return scores[-2], scores[-1]
    return None, None


# -----------------------------
# State
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "notes" not in st.session_state:
    st.session_state.notes = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "challenge_result" not in st.session_state:
    st.session_state.challenge_result = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## TeachBack AI")
    st.caption("Explain → find gaps → practise → improve")

    st.markdown("### Analysis mode")
    analysis_mode = st.radio(
        "Choose speed",
        ["Fast", "Deep analysis"],
        index=0,
        label_visibility="collapsed",
        help="Fast mode is instant. Deep AI uses Sentence-BERT and can take longer the first time.",
    )

    if analysis_mode == "Fast":
        st.success("Fast local analysis")
    else:
        st.info("Deeper semantic analysis. First use can take longer.")

    st.markdown("### Try a sample")
    sample_topic = st.selectbox("Sample topic", list(SAMPLES.keys()))

    if st.button("Load sample", use_container_width=True):
        st.session_state.notes = SAMPLES[sample_topic]
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.challenge_result = None
        st.rerun()

    if st.button("Clear progress", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.challenge_result = None
        st.rerun()

    st.divider()
    st.markdown(
        """
        **How it works**
        1. Add study notes
        2. AI extracts concepts
        3. Explain one yourself
        4. Gaps are detected
        5. Practice targets weaknesses
        6. Re-test to prove improvement
        """
    )

    st.caption("No paid API key required")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="topline">
        <h1>TeachBack AI</h1>
        <p>Test what you understand, find the gaps, and practise what needs work.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Built for students who want to test understanding, not just reread notes.")

notes = st.text_area(
    "Paste your study material",
    value=st.session_state.notes,
    height=190,
    placeholder="Paste lecture notes, textbook material, or revision notes here...",
)
st.session_state.notes = notes

if not clean_text(notes):
    st.info("👈 Load a sample from the sidebar, or paste your own notes to begin.")
    st.stop()

sample_name = next((name for name, text in SAMPLES.items() if clean_text(text) == clean_text(notes)), None)
if sample_name:
    concepts = CURATED_SAMPLE_CONCEPTS[sample_name]
    concept_pairs = [(c, 1.0) for c in concepts]
else:
    concept_pairs = extract_concepts(notes, top_n=8)
    concepts = [c for c, _ in concept_pairs]

if not concepts:
    st.error("I couldn't extract enough concepts. Add a little more text.")
    st.stop()

attempts = len(st.session_state.history)
avg = np.mean([h["score"] for h in st.session_state.history]) if attempts else None
mastered_count = len({
    h["concept"]
    for h in st.session_state.history
    if h["score"] >= 82
})

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Concepts found", len(concepts))
with c2:
    st.metric("Attempts", attempts)
with c3:
    st.metric("Average mastery", f"{avg:.0f}%" if avg is not None else "—")
with c4:
    st.metric("Mastered concepts", mastered_count)

st.markdown("#### Key concepts")
chips = "".join(
    f'<span class="concept-chip">{i+1}. {c.title()}</span>'
    for i, c in enumerate(concepts)
)
st.markdown(chips, unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(
    ["Teach-back", "Practice", "Progress"]
)

# -----------------------------
# Teach-back tab
# -----------------------------
with tab1:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### Teach it back")
        st.caption("Explain the idea as if you were teaching a friend.")

        selected = st.selectbox(
            "Choose a concept",
            concepts,
            format_func=lambda x: x.title()
        )

        context = relevant_context(notes, selected)

        with st.expander("Need a hint?"):
            hint_terms = important_terms(context, 4)
            if hint_terms:
                st.write("Think about:", ", ".join(x.title() for x in hint_terms))

        answer = st.text_area(
            "Your explanation",
            height=175,
            placeholder=f"Explain {selected} in simple words without copying your notes...",
            key="teach_answer",
        )

        analyse = st.button(
            "Analyse my understanding",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.markdown("### Feedback")

        if analyse:
            if len(answer.strip().split()) < 8:
                st.warning("Write at least a short explanation so the AI has enough to assess.")
            else:
                if analysis_mode == "Deep analysis":
                    with st.spinner("Running semantic analysis... first use may take a little longer."):
                        score, engine = semantic_score(context, answer, analysis_mode, selected)
                else:
                    score, engine = semantic_score(context, answer, analysis_mode, selected)

                raw_misses = missing_concepts(context, answer, concept=selected)
                misses = clean_missing_concepts(raw_misses)
                mismatch = detect_concept_mismatch(
                    notes, concepts, selected, answer
                )
                level, msg = feedback(score, misses)
                challenge_q, challenge_ref = diagnostic_question(
                    selected, misses, context
                )

                st.session_state.history.append({
                    "concept": selected,
                    "score": float(score),
                    "band": mastery_band(score),
                    "misses": misses,
                    "mismatch": mismatch["detected"] if mismatch else None,
                })

                st.session_state.last_result = {
                    "concept": selected,
                    "score": float(score),
                    "engine": engine,
                    "misses": misses,
                    "level": level,
                    "message": msg,
                    "context": context,
                    "answer": answer,
                    "mismatch": mismatch,
                    "challenge_question": challenge_q,
                    "challenge_reference": challenge_ref,
                    "challenge_weak_idea": misses[0] if misses else selected,
                }
                st.session_state.challenge_result = None

        result = st.session_state.last_result

        if result and result["concept"] == selected:
            score = result["score"]

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="small-muted">Understanding score</div>
                    <div class="result-score">{score:.0f}%</div>
                    <div style="margin-top:8px;font-weight:700;">{result['level']}</div>
                    <div class="small-muted" style="margin-top:6px;">{result['engine']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(100, int(round(score))) / 100)

            if score >= 82:
                st.success(result["message"])
            elif score >= 65:
                st.info(result["message"])
            else:
                st.warning(result["message"])

            mismatch = result.get("mismatch")
            if mismatch:
                st.error(
                    f"Possible concept mismatch: you selected **{mismatch['selected'].title()}**, "
                    f"but this explanation is mainly about **{mismatch['detected'].title()}**. "
                    "Switch the concept or rewrite the explanation for the selected topic."
                )

            if result["misses"]:
                st.markdown("**Missing or weak ideas**")
                st.write(" • ".join(x.title() for x in result["misses"]))

            st.markdown("#### Challenge my understanding")
            st.caption("One follow-up question is chosen from the gap in your explanation.")
            st.markdown(f"**{result['challenge_question']}**")

            challenge_answer = st.text_area(
                "Your answer to the challenge",
                height=105,
                key=f"challenge_{selected}",
                placeholder="Explain the reason in your own words..."
            )

            if st.button(
                "Check challenge answer",
                key=f"check_challenge_{selected}",
                use_container_width=True
            ):
                if len(challenge_answer.strip().split()) < 6:
                    st.warning("Give a short explanation, not just one or two words.")
                else:
                    challenge_score, _ = score_challenge_answer(
                        result["challenge_reference"],
                        challenge_answer,
                        result.get("challenge_weak_idea", selected),
                        analysis_mode,
                    )
                    st.session_state.challenge_result = {
                        "concept": selected,
                        "score": float(challenge_score),
                        "answer": challenge_answer,
                    }

            challenge_result = st.session_state.challenge_result
            if challenge_result and challenge_result.get("concept") == selected:
                cscore = challenge_result["score"]
                st.metric("Diagnostic challenge", f"{cscore:.0f}%")
                if cscore >= 75:
                    st.success(challenge_feedback(cscore))
                elif cscore >= 50:
                    st.info(challenge_feedback(cscore))
                else:
                    st.warning(challenge_feedback(cscore))

            previous, current = latest_two_scores(
                st.session_state.history,
                selected
            )

            if previous is not None:
                improvement = current - previous

                concept_attempts = [
                    h for h in st.session_state.history
                    if h["concept"] == selected
                ]
                earlier_gaps = len(concept_attempts[-2].get("misses", []))
                latest_gaps = len(concept_attempts[-1].get("misses", []))
                gaps_closed = max(0, earlier_gaps - latest_gaps)

                st.markdown(
                    f"""
                    <div style="font-weight:700; margin:14px 0 8px 0;">Progress on this concept</div>
                    <div class="progress-strip">
                        <div class="progress-cell">
                            <div class="progress-label">Earlier</div>
                            <div class="progress-number">{previous:.0f}%</div>
                        </div>
                        <div class="arrow">→</div>
                        <div class="progress-cell">
                            <div class="progress-label">Latest</div>
                            <div class="progress-number">{current:.0f}%</div>
                        </div>
                        <div class="arrow">→</div>
                        <div class="progress-cell">
                            <div class="progress-label">Change</div>
                            <div class="progress-number">{improvement:+.0f}%</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if improvement > 0 or gaps_closed > 0:
                    st.success(
                        f"Learning evidence: score changed by **{improvement:+.0f} points** "
                        f"and **{gaps_closed} knowledge gap{'s' if gaps_closed != 1 else ''}** "
                        "were resolved between the last two attempts."
                    )
            else:
                st.caption("First attempt saved. Practice the weak areas, then explain it again to reveal your improvement.")

            with st.expander("Source snippet to review"):
                st.write(result["context"])

        else:
            st.info("Your score, knowledge gaps, and improvement will appear here.")


# -----------------------------
# Adaptive practice tab
# -----------------------------
with tab2:
    st.markdown("### Targeted practice")
    st.caption("Questions are generated from your own notes and prioritised toward your latest gaps.")

    latest_missing = (
        st.session_state.history[-1].get("misses", [])
        if st.session_state.history else []
    )

    weak = [
        h["concept"]
        for h in st.session_state.history
        if h["score"] < 65
    ]

    focus_concepts = list(dict.fromkeys(
        latest_missing
        + weak
        + [c for c in concepts if c not in latest_missing and c not in weak]
    ))

    if latest_missing:
        st.success(
            "Adaptive focus: "
            + ", ".join(x.title() for x in latest_missing)
        )
    elif weak:
        st.info(
            "Practice prioritised toward: "
            + ", ".join(dict.fromkeys(x.title() for x in weak))
        )
    else:
        st.info("Complete one teach-back attempt and this section will automatically target your weak areas.")

    qs = make_cloze_questions(notes, focus_concepts, n=5)

    for i, (q, ans) in enumerate(qs, 1):
        with st.container(border=True):
            st.markdown(f"**Question {i}**")
            st.write(q)

            user_q = st.text_input(
                "Your answer",
                key=f"q_{i}_{abs(hash(q))}",
                placeholder="Type the missing idea...",
            )

            if user_q:
                normalized_user = re.sub(r"\W+", " ", user_q.lower()).strip()
                normalized_ans = re.sub(r"\W+", " ", ans.lower()).strip()

                if normalized_ans in normalized_user or normalized_user in normalized_ans:
                    st.success("Correct ✓")
                else:
                    st.warning(f"Not quite — review this idea: **{ans.title()}**")


# -----------------------------
# Dashboard tab
# -----------------------------
with tab3:
    st.markdown("### Your mastery dashboard")

    if not st.session_state.history:
        st.info("Complete at least one teach-back attempt to start tracking progress.")
    else:
        df = pd.DataFrame(st.session_state.history).copy()
        df["Attempt"] = range(1, len(df) + 1)
        df["Concept"] = df["concept"].str.title()

        top1, top2, top3 = st.columns(3)

        with top1:
            st.metric("Total attempts", len(df))

        with top2:
            latest_score = df.iloc[-1]["score"]
            st.metric("Latest score", f"{latest_score:.0f}%")

        with top3:
            first_score = df.iloc[0]["score"]
            overall_gain = latest_score - first_score
            st.metric(
                "Overall change",
                f"{overall_gain:+.0f}%"
            )

        st.markdown("#### Progress over time")
        chart_df = df[["Attempt", "score"]].set_index("Attempt")
        st.line_chart(chart_df, height=270)

        st.markdown("#### Mastery by concept")
        summary = (
            df.groupby("Concept", as_index=False)["score"]
              .mean()
              .sort_values("score")
        )
        summary["Status"] = summary["score"].apply(mastery_band)

        st.bar_chart(
            summary.set_index("Concept")["score"],
            horizontal=True,
            height=300
        )

        display_summary = summary.rename(
            columns={"score": "Average score"}
        )
        display_summary["Average score"] = display_summary["Average score"].round(0).astype(int).astype(str) + "%"

        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True,
        )

        lowest = summary.iloc[0]
        st.warning(
            f"Best next step: revisit **{lowest['Concept']}**. "
            f"Current average mastery: **{lowest['score']:.0f}%**."
        )

        # Strong demo moment for repeated attempts on the same concept.
        repeated = (
            df.groupby("Concept")
              .filter(lambda g: len(g) >= 2)
        )

        if not repeated.empty:
            latest_concept = repeated.iloc[-1]["Concept"]
            concept_rows = df[df["Concept"] == latest_concept]
            before = concept_rows.iloc[-2]["score"]
            after = concept_rows.iloc[-1]["score"]
            gain = after - before

            st.markdown(
                f"""
                <div style="font-weight:700; margin:14px 0 8px 0;">Progress: {latest_concept}</div>
                <div class="progress-strip">
                    <div class="progress-cell">
                        <div class="progress-label">Earlier attempt</div>
                        <div class="progress-number">{before:.0f}%</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="progress-cell">
                        <div class="progress-label">Latest attempt</div>
                        <div class="progress-number">{after:.0f}%</div>
                    </div>
                    <div class="arrow">→</div>
                    <div class="progress-cell">
                        <div class="progress-label">Change</div>
                        <div class="progress-number">{gain:+.0f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "TeachBack AI is a learning aid, not an official grading system. "
    "Scores guide revision and are not formal academic marks."
)

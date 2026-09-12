# TeachBack AI 🧠

**Turn passive notes into active understanding.**

TeachBack AI is an adaptive study coach that uses NLP and semantic similarity to test whether a learner can explain a concept in their own words. Instead of rewarding recognition or memorisation, it uses the teach-back/Feynman approach: learn something, explain it simply, identify gaps, then practise the weak areas.

## Why this project exists

Students often reread notes and feel familiar with the material without being able to explain it independently. TeachBack AI converts any set of notes into an active learning loop:

1. Extract important concepts from the learner's own notes.
2. Ask the learner to explain one concept in plain language.
3. Compare the explanation with the most relevant source context using semantic embeddings.
4. Identify missing concepts and give targeted feedback.
5. Generate adaptive practice that prioritises weaker topics.
6. Track mastery over repeated attempts.

## AI/ML used

- **Sentence-BERT (`all-MiniLM-L6-v2`)** for semantic similarity between the learner's explanation and source material.
- **TF-IDF n-gram extraction** for concept discovery.
- **Cosine similarity** for context retrieval and a fallback assessment engine.
- **Adaptive sequencing** that prioritises topics with lower mastery scores.

The AI is core to the learning loop rather than a decorative chatbot feature.

## Features

- Paste any lecture notes or textbook section
- Automatic concept extraction
- Teach-back assessment in the student's own words
- Semantic understanding score
- Missing-concept detection
- Adaptive cloze practice
- Mastery dashboard and revision recommendation
- No paid API key required
- Streamlit interface for fast, accessible web deployment

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The Sentence-BERT model downloads automatically on first run. If it cannot load, the app falls back to TF-IDF cosine similarity so the demo still works.

## Suggested Devpost tagline

> An adaptive AI study coach that finds out what you can actually explain, detects knowledge gaps, and turns your own notes into targeted practice.

## Judging fit

### Educational Impact
Targets a real learning problem: the illusion of competence caused by passive rereading. It encourages retrieval practice and explanation rather than copying.

### Creative Use of AI/ML
Uses semantic embeddings to assess meaning, NLP to discover concepts, and mastery data to adapt the learner's next practice.

### Technical Execution
Functional end-to-end Streamlit app, graceful model fallback, clear feedback, adaptive state, and visual mastery tracking.

### Pitch & Demo
The value is visible in one flow: paste notes → explain → AI finds a missing idea → adaptive practice changes.

## Important hackathon note

The Virgo rules page currently contains conflicting date wording for the coding window. Keep your Git commit history showing that this project was created during the current challenge period, and ask the organiser for clarification if needed.

## Updated competition build
This build includes curated concept feedback for the built-in demos, cleaner missing-idea detection, concept-mismatch checks, synonym-aware scoring, a diagnostic challenge, adaptive practice, and progress tracking. Fast mode remains the recommended demo mode because it works locally without a paid API key.

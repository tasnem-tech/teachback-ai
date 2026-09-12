# Devpost submission draft

## Project name
TeachBack AI

## Tagline
An adaptive AI study coach that tests what you can explain, finds knowledge gaps, and turns your own notes into targeted practice.

## Inspiration
Rereading notes can make a topic feel familiar without proving that we understand it. A much harder test is simple: can you explain the idea clearly without looking at the answer? I wanted to turn that teach-back method into a personalised AI learning loop that works with any student's own material.

## What it does
TeachBack AI lets a learner paste lecture notes or textbook material. It automatically extracts the main concepts and asks the learner to explain one in their own words. A semantic NLP model compares the learner's explanation with the most relevant source context, estimates understanding, identifies missing ideas, and gives focused feedback. The learner then receives adaptive practice that prioritises weak concepts. A mastery dashboard shows what to revise next.

## How I built it
I built the interface in Streamlit and the learning pipeline in Python. TF-IDF n-grams extract important concepts from the source material. Context retrieval uses cosine similarity. Sentence-BERT embeddings (`all-MiniLM-L6-v2`) compare the meaning of the student's explanation with the source context. The app then detects uncovered concepts and stores mastery scores so later practice can prioritise weaker areas. A TF-IDF fallback keeps the application functional if the embedding model is unavailable.

## Challenges I ran into
The biggest challenge was making the feedback useful without pretending that an AI similarity score is an official academic grade. I therefore treat the score as a revision signal, combine it with missing-concept detection, and explicitly label the system as a learning aid rather than a grading tool.

## Accomplishments that I'm proud of
- Built a complete active-learning loop rather than a generic chatbot.
- Made AI central to the educational interaction.
- Added adaptive practice based on the learner's previous weak areas.
- Designed the app to run without a paid AI API key.
- Added a fallback NLP engine to make the demo more reliable.

## What I learned
I learned how semantic embeddings can be applied to educational feedback, but also why similarity alone should not be treated as a ground-truth grade. The most useful design combines semantic comparison, concept coverage, and transparent feedback.

## What's next for TeachBack AI
Next I would add PDF upload, multilingual teach-back, speech-to-text explanations, spaced repetition across study sessions, and teacher-created mastery rubrics.

## Built with
Python, Streamlit, scikit-learn, Sentence-Transformers, Sentence-BERT, pandas, NumPy

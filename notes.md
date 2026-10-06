You create or open the .db file in your notebook, connect with sqlite3.connect, then use your Python functions to execute SQL on that connection. That's the workflow.



PHASE 1

BANKING77

&#x20;   ↓

NLP baseline

&#x20;   ↓

TF-IDF + Logistic Regression



&#x20;       ↓



PHASE 2

DistilBERT

&#x20;   ↓

Fine-tuned intent classifier



&#x20;       ↓



PHASE 3

Banking Knowledge Base

&#x20;   ↓

Embeddings

&#x20;   ↓

ChromaDB

&#x20;   ↓

Retriever

&#x20;   ↓

LLM

&#x20;   ↓

RAG



&#x20;       ↓



PHASE 4

Agent

&#x20;   ↓

RAG tool

&#x20;   ↓

Transaction tool

&#x20;   ↓

Calculator

&#x20;   ↓

Policy tool



&#x20;       ↓



PHASE 5

MCP



&#x20;       ↓



PHASE 6

FastAPI

&#x20;   ↓

Tests

&#x20;   ↓

Docker

&#x20;   ↓

Deployment

&#x20;   ↓









github\_pat\_11A37BHDQ0Tt7TRFm943ko\_JGikGq3VJZ12nC0EiIxFYyUJI6tw7Zg9xts0fR1sbWlKXG4C5BC4HlNVVka

Evaluation / logging



Notes:



TF-IDF has formulas to calculate the TF-IDF weight, and in your TfidfVectorizer() setup, scikit-learn then normalizes the resulting vector internally by default.



Statistical = based on patterns and frequencies observed in the data.

* "TF-IDF represents text using statistical word-occurrence patterns, and the classifier learns which patterns are associated with each intent."
* 
* That's accurate and sounds much stronger.
* And this is one reason DistilBERT is different: it can learn contextual relationships between words, rather than relying mainly on these word-occurrence statistics.
* Pattern means a repeated or recognizable way that things occur or are related.
* In our ML context:
* A statistical pattern = a relationship in the data that appears repeatedly.
* Related means connected to or associated with something.






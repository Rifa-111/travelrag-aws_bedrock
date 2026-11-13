# TravelRAG — Retrieval-Augmented Travel Policy Assistant

A small **portfolio demonstration** of document retrieval, evidence inspection and optional LLM answer generation with **Amazon Bedrock**. Uses **fictional** policies.

## Architecture

`Streamlit question → TF-IDF chunk retrieval → relevance threshold → (optional) Bedrock Converse → answer + retrieved evidence`

This project uses **local scikit-learn TF-IDF retrieval**, not Bedrock Knowledge Bases, vector embeddings or SageMaker. Do not describe it as experience with those services. It demonstrates a working integration path for Bedrock Converse, subject to AWS account access and model availability.

## Run locally (no AWS charges)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ask: **What is the hotel reimbursement limit?** In the default free mode, the app displays matching evidence rather than inventing a generated answer.

## Enable Bedrock (optional, may incur AWS charges)

1. Configure AWS credentials via an AWS profile, environment variables or IAM role. **Never commit credentials.**
2. Ensure the selected foundation model is enabled and supports the Bedrock Converse API in your chosen region. Check current model availability and IAM permissions.
3. Set `AWS_REGION=eu-west-2` (or your region) and `BEDROCK_MODEL_ID=<your-supported-model-id>`.
4. Start the app and ask a question. Bedrock usage is billed by AWS according to model and usage; set budgets/alerts and stop using it when finished.

Required IAM permission for the selected model typically includes `bedrock:InvokeModel`. Use least-privilege access scoped to the permitted model ARN.

## Guardrails and limitations

- A minimum retrieval-similarity threshold causes abstention for unrelated queries.
- The system prompt instructs the model to use only retrieved evidence, cite source numbers and ignore instructions embedded in documents.
- This is **prompt-level grounding**, not a deployed Amazon Bedrock Guardrails policy or a guarantee against prompt injection/hallucinations.
- Source excerpts are displayed for manual verification. Generated citation correctness is **not** automatically validated.
- No PII, authentication, persistent logging, production monitoring, evaluation benchmark, managed feature store, SageMaker or production deployment is included.
- TF-IDF is cheap and simple but lacks semantic understanding. A sensible next step is optional Bedrock embeddings plus a local vector index and a groundedness evaluation set.

## Tests

```bash
python -m pytest -q
```

GitHub Actions runs the unit tests without AWS credentials. Bedrock integration testing requires a configured account.

#
> Built a Python/Streamlit travel-policy RAG prototype with local TF-IDF retrieval, source inspection and optional Amazon Bedrock Converse integration; added relevance-based abstention and automated retrieval tests.

If you have not executed a successful Bedrock call, describe the integration as **implemented but not cloud-validated**. Do not claim SageMaker experience from this project.

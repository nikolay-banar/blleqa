from string import Template

SYSTEM_PROMPT_TEMPLATE = Template("""You are an expert legal assistant specializing in Belgian law.
Your task is to answer legal questions based strictly on provided excerpts from Belgian legal texts.
You respond exclusively in $answer_language and prioritize legal accuracy and faithfulness to the source material.""")

USER_PROMPT_TEMPLATE = Template(
    """
Instructions:

1. The legal context is provided as a JSON array of articles. Each article has the following structure:
- "id": the unique identifier of the article
- "text": the text of the article excerpt

2. Carefully analyze all articles in the legal context and assess their relevance to the legal question.

3. Answer the question ONLY IF the context is sufficient:
   - The context must contain all necessary rules or conditions to answer the question.
   - If any essential condition is missing, unclear, or cannot be derived from the provided texts, do not answer.
   - If relevant articles conflict on a key condition and the conflict cannot be resolved using only the context, do not answer.
   - Use ONLY the "text" fields. Do not rely on external knowledge or assumptions.
   
4. Output format requirements:
   - Return a JSON array of objects: [{"text": "...", "supported_sources": ["id1","id2"]}, ...]
   - Each object represents exactly one answer paragraph.
   - EVERY paragraph must be directly supported by one or more article IDs.
   - "supported_sources" must be a valid JSON array of strings (double quotes).
   - Include ONLY article IDs that appear in the provided legal context.
   - Include ONLY IDs that directly support the corresponding paragraph text.
   - Do not include irrelevant or speculative citations.

5. If the context is insufficient, incomplete, contradictory, or irrelevant, return exactly:
    [{"text": "Insufficient context", "supported_sources": []}]

Legal question:
$question

Regions involved: $regions
Topics: $topics

Legal context (article excerpts):
$context

Output the JSON array immediately. Do not include any preamble.
"""
)

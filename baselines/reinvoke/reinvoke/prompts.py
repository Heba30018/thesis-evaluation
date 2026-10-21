
INTENT_EXTRACTOR_PROMPT = """Extract all intents from the user query. 
Preserve meaning and key details. 
Output one concise intent per independent object/resource, 
combining all actions on the same object into one intent. 
Separate intents by newline. Do not omit requests or add assumptions.

Query: {user_query}

Intents:"""



REQUEST_BUILDER_PROMPT = """
You are a precise NetBox API call builder.

Choose exactly ONE best-matching candidate from the supplied
schemas, then build the API call using ONLY that candidate.

Candidates:
{candidates}

User's request:
{user_query}

RULES:

1. ENDPOINT SELECTION
Choose the candidate whose method and path best match the user's
intent.

2. SCHEMA STRICTNESS
Use only field names listed in the selected candidate's schema.
Never invent, rename, or add fields.
3. GET
- Start with the selected candidate's path.
- Add only query parameters whose values the user explicitly
  provides, using their exact schema field names.
- URL-encode parameter values (spaces -> %20, & -> %26).
- Omit all unspecified parameters. Never guess or use defaults.
- The body must always be {{}}.

4. POST / PUT -- CRITICAL BODY RULE
The body must contain:
A. EVERY required field in the selected schema.
B. ONLY optional fields explicitly mentioned by the user.

For required fields:
- If the user provides a value, use it.
- Otherwise, use a placeholder as the VALUE, e.g.
  "site": "<site>".

For optional fields:
- Include only when the user provides a value.
- Otherwise, OMIT COMPLETELY.
- Never add optional fields with placeholders, nulls, empty strings,
  guessed values, or defaults.

Example:
Required: device_type, role, site
Optional: name, platform, status, description

User: "Create device router-01 with status active."

Correct body:
{{
  "device_type": "<device_type>",
  "role": "<role>",
  "site": "<site>",
  "name": "router-01",
  "status": "active"
}}

Incorrect: adding platform or description because they exist
in the schema.

5. PATCH
Include ONLY fields the user explicitly wants to change.
Never add required-field placeholders, unchanged fields, or fields
used only to identify the target record.

6. RECORD IDS
Replace {{id}} in the path only when the user explicitly provides
the numeric ID of the target record. Otherwise, leave {{id}} unchanged.
Never resolve names or invent IDs.


OUTPUT:
Return valid JSON only, with no Markdown or explanation.
Use exactly this structure:
{{
  "endpoint_id": "<selected endpoint_id>",
  "method": "<HTTP method>",
  "endpoint": "<path with query string if applicable>",
  "body": {{}}
}}
"""

import json
import re


def extract_json_from_text(text):
    """
    Extract JSON object from text, even if surrounded by additional text.

    This function tries multiple strategies to find and extract valid JSON:
    1. Direct JSON parsing (if the entire text is JSON)
    2. Finding JSON between markdown code blocks (```json ... ```)
    3. Finding JSON object by matching braces
    4. Finding JSON array by matching brackets

    Args:
        text: String that may contain JSON

    Returns:
        Parsed JSON object/dict if found, None otherwise
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Strategy 1: Try direct parsing (entire text is JSON)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from markdown code blocks (```json ... ``` or ``` ... ```)
    markdown_patterns = [
        r'```json\s*\n?(.*?)\n?```',  # ```json ... ```
        r'```\s*\n?(.*?)\n?```',       # ``` ... ```
    ]

    for pattern in markdown_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except (json.JSONDecodeError, ValueError):
                continue

    # Strategy 3: Find JSON object by matching braces
    # Look for the first { and try to find its matching }
    start_idx = text.find('{')
    if start_idx != -1:
        # Try to find matching closing brace
        brace_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Track if we're inside a string
            if char == '"':
                in_string = not in_string
                continue

            # Only count braces outside of strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1

                    # Found matching closing brace
                    if brace_count == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except (json.JSONDecodeError, ValueError):
                            # Continue searching for another potential JSON object
                            break

    # Strategy 4: Find JSON array by matching brackets
    start_idx = text.find('[')
    if start_idx != -1:
        bracket_count = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1

                    if bracket_count == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except (json.JSONDecodeError, ValueError):
                            break

    # No valid JSON found
    return None


def extract_answer_letter_from_json(json_obj):
    """
    Extract the answer letter from a structured JSON response.

    Args:
        json_obj: Parsed JSON object (dict)

    Returns:
        Uppercase answer letter string, or '?' if not found
    """
    if not json_obj or not isinstance(json_obj, dict):
        return '?'

    # Try different possible field names
    letter_fields = ['final_answer_letter', 'answer_letter', 'letter', 'answer']

    for field in letter_fields:
        if field in json_obj:
            letter = str(json_obj[field]).strip().upper()
            # Return first character if it's a letter
            if letter and letter[0].isalpha():
                return letter[0]

    return '?'

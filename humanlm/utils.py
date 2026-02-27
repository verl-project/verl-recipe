# Copyright 2026 HUMANLM team and/or its affiliates
# Copyright 2026 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

# Only match <think>...</think>
NON_RESPONSE_PAIR_RE = re.compile(
    r"<\s*think\b[^>]*>.*?</\s*think\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Only strip leading junk that ends with </think>
NON_RESPONSE_CLOSE_PREFIX_RE = re.compile(
    r"^.*?</\s*think\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Only strip leftover <think ...> openings
NON_RESPONSE_OPEN_RE = re.compile(
    r"<\s*think\b[^>]*>",
    re.IGNORECASE,
)

"""
Remove <tag> ... </tag> blocks where tag != 'response'
Remove Any leading text ending at a non-response </tag> if there's no matching <tag> before it
remove any leftover non-response <tag ...> openings
      - Any <tag> ... </tag> blocks where tag != 'response'
      - Any leading text ending at a non-response </tag> if there's
        no matching <tag> before it
"""
def _strip_non_response_tags(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = NON_RESPONSE_PAIR_RE.sub("", text)
    text = NON_RESPONSE_CLOSE_PREFIX_RE.sub("", text)
    text = NON_RESPONSE_OPEN_RE.sub("", text)

    return text

"""
    First remove anything of the form <tag> ... <\tag> where tag is not "response"
    Extract text in <response>...</response>, or just <response>..., or even just ... <\response> fallback to full text
"""
def extract_response(text: str) -> str:
    # remove all tags except <response> ... </response>
    text = _strip_non_response_tags(text)
    original_test_no_tag = text

    #get <response> ... </response>
    pattern = re.compile(
        r"<\s*response\s*>(.*?)</\s*response\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    #<response> but no closing tag, get everything after <response>
    open_pattern = re.compile(r"<\s*response\s*>", re.IGNORECASE)
    open_match = open_pattern.search(text)
    if open_match:
        if len(text[open_match.end():].strip()) == 0:
            return original_test_no_tag
        return text[open_match.end():].strip()

    #no <response>, but there is </response>, then get everything before </response>
    close_pattern = re.compile(r"</\s*response\s*>", re.IGNORECASE)
    close_match = close_pattern.search(text)
    if close_match:
        return text[:close_match.start()].strip()
    return text.strip()


import json
def parse_messages(messages, strip_sys_prompt=True):
    """
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'},
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    """
    if messages is None:
        return ""

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = "\n".join(f"**{m['role'].capitalize()} {m.get('name')}**: {m['content']}" for m in messages)

    return chat


def strip_system_prompt(messages):
    """
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'},
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    """
    return [msg for msg in messages if msg['role'] != "system"]

def parse_messages_preprocessing(messages, strip_sys_prompt=True):
    """
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'},
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    """
    if messages is None:
        return ""

    def strip_system_prompt(messages):
        return [msg for msg in messages if msg["role"] != "system"]

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = "\n".join(f"Message from {m['role']}: \n'''\n{m['content']}\n'''" for m in messages)

    return chat


# Should we say "article title" instead?
# Role is None? 
def parse_messages_title_preprocessing(messages, strip_sys_prompt=True):
    """
    For Medium-style threads:
      - For the first non-system message: use (title, subtitle) from metadata.
      - For remaining messages: use the regular message content (like parse_messages).
    All joined into a single context string.
    """
    if messages is None:
        return ""

    def strip_system_prompt(msgs):
        return [m for m in msgs if m.get("role") != "system"]

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    if not messages:
        return ""

    def get_meta_dict(m):
        meta = m.get("metadata")
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, str):
            meta = meta.strip()
            if not meta:
                return {}
            try:
                return json.loads(meta)
            except json.JSONDecodeError:
                return {}
        return {}

    first = messages[0]
    meta = get_meta_dict(first)

    counts = meta.get("counts", {}) or {}
    title = counts.get("title") or meta.get("title") or ""
    subtitle = counts.get("subtitle") or meta.get("subtitle") or ""

    role = first.get("role", "unknown")
    lines = [
        f"Article from {role}: \n'''\nTitle: {title}\nSubtitle: {subtitle}\n'''"
    ]

    rest = messages[1:]
    if rest:
        # Reuse existing parse_messages, but don't strip system again
        rest_str = parse_messages(rest, strip_sys_prompt=False)
        if rest_str:
            lines.append(rest_str)

    return "\n".join(lines)


def extract_json(s: str):
    """
    Best-effort extractor for the JSON object the model is supposed to return.

    1. Try the full custom JSON-ish parser (supports triple-quoted strings).
    2. If that fails, fall back to a regex-based extraction of `"score": <number>`.
    """

    # -------------------------------
    # 1. Your existing parser logic
    # -------------------------------
    def convert_value(value):
        true_values = {"true": True, "false": False, "null": None}
        value_lower = value.lower()
        if value_lower in true_values:
            return true_values[value_lower]
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as string if not a number

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in "-+0123456789.eE":
            pos += 1
        num_str = s[start:pos]
        try:
            if "." in num_str or "e" in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            raise ValueError(f"Invalid number at position {start}: {num_str}")

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ""
        while pos < len(s):
            c = s[pos]
            if c == "\\":
                pos += 1
                if pos >= len(s):
                    raise ValueError("Invalid escape sequence")
                c = s[pos]
                escape_sequences = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", quote_char: quote_char}
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += c
            pos += 1
        raise ValueError("Unterminated string")

    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f"Expected string for key at position {pos}")

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == "{"
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != ":":
                raise ValueError(f'Expected ":" at position {pos}')
            pos += 1
            pos = skip_whitespace(s, pos)
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "}":
                break
            elif pos < len(s) and s[pos] != "}":
                raise ValueError(f'Expected "," or "}}" at position {pos}')
        if pos >= len(s) or s[pos] != "}":
            raise ValueError(f'Expected "}}" at position {pos}')
        pos += 1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == "["
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "]":
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "]":
                break
            elif pos < len(s) and s[pos] != "]":
                raise ValueError(f'Expected "," or "]" at position {pos}')
        if pos >= len(s) or s[pos] != "]":
            raise ValueError(f'Expected "]" at position {pos}')
        pos += 1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos : pos + 3] == "'''":
            quote_str = "'''"
        elif s[pos : pos + 3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f"Expected triple quotes at position {pos}")
        pos += 3
        result = ""
        while pos < len(s):
            if s[pos : pos + 3] == quote_str:
                pos += 3
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += s[pos]
                pos += 1
        raise ValueError("Unterminated triple-quoted string")

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        if pos >= len(s):
            raise ValueError("Unexpected end of input")
        if s[pos] == "{":
            return parse_object(s, pos)
        elif s[pos] == "[":
            return parse_array(s, pos)
        elif s[pos : pos + 3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos : pos + 4].lower() == "true":
            return True, pos + 4
        elif s[pos : pos + 5].lower() == "false":
            return False, pos + 5
        elif s[pos : pos + 4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

    # Try the full parse first
    try:
        json_start = s.index("{")
        json_end = s.rfind("}")
        sub = s[json_start : json_end + 1].strip()

        result, pos = parse_value(sub, 0)
        pos = skip_whitespace(sub, pos)
        # If there's trailing junk after the first object, ignore it but don't crash.
        # (We only care about "score" anyway.)
        # if pos != len(sub):
        #     raise ValueError(f"Unexpected content at position {pos}")
        return result
    except Exception:
        # -------------------------------
        # 2. Fallback: regex "score": <num>
        # -------------------------------
        m = re.search(r'"score"\s*:\s*([-+]?\d*\.?\d+)', s)
        if m:
            try:
                score = float(m.group(1))
            except ValueError:
                score = 0.0
            return {"score": score}

        # If even regex fails, bubble up a generic error
        raise

"""Comment/code separation and input formatting for multitask experiments."""

from __future__ import annotations

import re
from typing import Literal, Tuple

# C/C++ style comments dominate MADE-WIC / this competition corpus.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
# Keep # only when it looks like a comment, not a preprocessor directive alone.
_HASH_COMMENT = re.compile(r"(?m)^\s*#(?!\s*(include|define|if|ifdef|ifndef|endif|pragma|undef|elif|else)\b).*?$")


def extract_comments_and_code(function_text: str) -> Tuple[str, str]:
    """Split a formatted function into (comments, code_without_comments)."""
    text = function_text or ""
    comments: list[str] = []

    def _keep_block(match: re.Match[str]) -> str:
        comments.append(match.group(0).strip())
        return " "

    def _keep_line(match: re.Match[str]) -> str:
        comments.append(match.group(0).strip())
        return " "

    without_block = _BLOCK_COMMENT.sub(_keep_block, text)
    without_line = _LINE_COMMENT.sub(_keep_line, without_block)
    # Do not strip preprocessor directives; only capture non-directive # lines.
    code = _HASH_COMMENT.sub(_keep_line, without_line)
    code = re.sub(r"[ \t]+\n", "\n", code)
    code = re.sub(r"\n{3,}", "\n\n", code).strip()
    comment_text = "\n".join(c for c in comments if c).strip()
    return comment_text, code


FormatMode = Literal["raw", "bimodal", "comments_only", "code_only"]


def format_input(
    function_text: str,
    mode: FormatMode = "raw",
    sep_token: str = "</s>",
) -> str:
    """
    Build model input text.

    - raw: original Function string (official baseline)
    - bimodal: comments then code, RoBERTa/CodeBERT style with </s>
    - comments_only / code_only: ablation modes
    """
    if mode == "raw":
        return function_text or ""

    comments, code = extract_comments_and_code(function_text)
    if mode == "comments_only":
        return comments
    if mode == "code_only":
        return code
    # bimodal
    left = comments if comments else ""
    right = code if code else (function_text or "")
    if left and right:
        return f"{left}{sep_token}{right}"
    return left or right

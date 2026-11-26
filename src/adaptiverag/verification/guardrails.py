"""Deterministic ($0, ~0 ms) input safety guardrails: prompt-injection & PII protection.

Screens incoming user queries and chat history before they reach external LLM APIs:
1. High-precision EN + TR prompt-injection, jailbreak, and system-prompt exfiltration rules
   (with NFKC + zero-width character normalization to resist obfuscation).
2. Checksum-validated PII & secret detection/redaction (API keys/tokens, email addresses,
   Luhn-verified credit card numbers, Turkish TC Kimlik No checksum-verified IDs, and
   international/Turkish phone numbers).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from adaptiverag.schemas.query import ChatMessage

PiiMode = Literal["off", "redact", "block"]
PiiEntityType = Literal["api_key", "email", "credit_card", "tr_national_id", "phone"]

# Strip invisible formatting / bidi control characters that attackers use to evade regex rules.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")
_ROLE_SPOOF_RE = re.compile(
    r"(?:<\|im_start\|>\s*(?:system|developer|assistant)|<\|im_end\|>|<\|system\|>"
    r"|\[SYSTEM\s+OVERRIDE\]|###\s*SYSTEM\s+INSTRUCTIONS?\s*:)",
    re.IGNORECASE,
)


class GuardrailBlockedError(ValueError):
    """Raised when deterministic input safety guardrails reject a query or chat history."""

    def __init__(self, decision: GuardrailDecision) -> None:
        self.decision = decision
        super().__init__(decision.blocked_reason or "Request blocked by input safety guardrails.")


@dataclass(frozen=True)
class _InjectionRule:
    rule_id: str
    pattern: re.Pattern[str]


# High-precision rules designed for zero false positives on legitimate technical AI/RAG questions
# (e.g., "How does Langfuse manage system prompts?") while catching explicit adversarial directives.
_INJECTION_RULES: tuple[_InjectionRule, ...] = (
    _InjectionRule(
        rule_id="instruction_override_en",
        pattern=re.compile(
            r"\b(?:"
            r"(?:ignore|disregard|override|bypass|forget|drop)\s+"
            r"(?:all\s+|any\s+|every\s+|your\s+|the\s+)?"
            r"(?:previous|prior|above|earlier|preceding|system|developer|original)\s+"
            r"(?:instructions?|prompts?|rules?|directives?|guidelines?|constraints?)"
            r"|forget\s+everything\s+(?:above|before|previously|you\s+were\s+told)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    _InjectionRule(
        rule_id="instruction_override_tr",
        pattern=re.compile(
            r"(?:"
            r"(?:önceki|yukarıdaki|mevcut|tüm|sistem|geliştirici)\s+"
            r"(?:tüm\s+|bütün\s+)?"
            r"(?:talimat(?:lar)?[ıiuü]?n?[ıiuü]?|kural(?:lar)?[ıiuü]?n?[ıiuü]?|yönerge(?:ler)?[iı]?n?[iı]?|prompt(?:lar)?[ıiuü]?n?[ıiuü]?)\s+"
            r"(?:yok\s*say|unut|görmezden\s*gel|iptal\s*et|devre\s*dışı\s*bırak|atla)"
            r"|(?:yok\s*say|unut|görmezden\s*gel).{0,24}(?:önceki|sistem)\s+(?:talimat|kural|yönerge)"
            r")",
            re.IGNORECASE,
        ),
    ),
    _InjectionRule(
        rule_id="system_prompt_exfiltration_en",
        pattern=re.compile(
            r"\b(?:reveal|print|show|output|dump|display|repeat|leak|expose|return|give\s+me|tell\s+me)\s+"
            r"(?:me\s+)?"
            r"(?:your|the)\s+"
            r"(?:hidden\s+|exact\s+|full\s+|complete\s+|original\s+|verbatim\s+|raw\s+|internal\s+)?"
            r"(?:system\s+prompt|developer\s+prompt|hidden\s+instructions?|initial\s+instructions?|secret\s+instructions?)\b",
            re.IGNORECASE,
        ),
    ),
    _InjectionRule(
        rule_id="system_prompt_exfiltration_tr",
        pattern=re.compile(
            r"(?:sistem\s+(?:prompt(?:unu|u|larını)?|istemi(?:ni)?|talimat(?:lar)?[ıi]n?[ıi]?)"
            r"|gizli\s+(?:talimat(?:lar)?[ıi]n?[ıi]?|prompt(?:unu|u)?))\s+"
            r"(?:göster|yazdır|yaz|paylaş|söyle|tekrarla|dökümle|ver|açıkla)",
            re.IGNORECASE,
        ),
    ),
    _InjectionRule(
        rule_id="jailbreak_mode_en",
        pattern=re.compile(
            r"\b(?:"
            r"(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|enter|switch\s+to|enable)\s+"
            r"(?:a\s+|an\s+|the\s+)?"
            r"(?:DAN\b|do\s+anything\s+now|developer\s+mode|unrestricted\s+mode|jailbreak(?:ed)?\s+mode|god\s+mode)"
            r"|(?:disable|bypass|turn\s+off|remove)\s+(?:all\s+|your\s+)?(?:safety|security|content)\s+"
            r"(?:filters?|guardrails?|restrictions?|policies|checks?)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    _InjectionRule(
        rule_id="jailbreak_mode_tr",
        pattern=re.compile(
            r"(?:güvenlik\s+(?:filtre(?:lerini|leri)|kuralları(?:nı)?|sınırları(?:nı)?|kontrollerini)"
            r"|kısıtlamaları(?:nı)?)\s+"
            r"(?:atla|aş|kaldır|devre\s*dışı\s*bırak|kapat)",
            re.IGNORECASE,
        ),
    ),
    _InjectionRule(
        rule_id="role_delimiter_spoofing",
        pattern=re.compile(
            r"(?:<\|im_start\|>\s*system|<\|system\|>|\[SYSTEM\s+OVERRIDE\]|###\s*SYSTEM\s+INSTRUCTIONS?\s*:)",
            re.IGNORECASE,
        ),
    ),
)

# Secret / API Key patterns (high entropy vendor prefixes)
_API_KEY_RE = re.compile(
    r"\b(?:"
    r"sk-or-v1-[A-Za-z0-9_-]{16,}"
    r"|sk-(?:proj-|ant-api\d+-|live-|test-)?[A-Za-z0-9_-]{20,}"
    r"|(?:pk|sk)-lf-[A-Za-z0-9_-]{12,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r")\b"
)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Candidate credit-card spans (13-19 digits with optional spaces/hyphens), validated via Luhn.
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

# Candidate 11-digit Turkish TC Kimlik No, validated via official mod-10 checksum.
_TCKN_CANDIDATE_RE = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")

# International & Turkish/US phone numbers (kept specific to avoid matching version numbers or dates).
_PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    r"\+90[\s.-]?\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}"
    r"|05\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}"
    r"|\+[1-9]\d{0,2}[\s.-]\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4}"
    r"|(?:\+1[\s.-]?)?\(\d{3}\)[\s.-]?\d{3}[\s.-]\d{4}"
    r")(?!\w)"
)


class InputGuardrailConfig(BaseModel):
    """Runtime policy for deterministic input guardrails."""

    enabled: bool = True
    block_prompt_injection: bool = True
    pii_mode: PiiMode = "redact"


class PiiFinding(BaseModel):
    """Summary of a detected PII or secret category (never stores raw secret values)."""

    entity_type: PiiEntityType
    placeholder: str
    count: int = Field(ge=1)


class GuardrailDecision(BaseModel):
    """Structured outcome from screening a user query, chat history, and thread instructions."""

    allowed: bool = True
    sanitized_query: str
    sanitized_history: list[ChatMessage] = Field(default_factory=list)
    sanitized_custom_instructions: str | None = None
    blocked_reason: str | None = None
    triggered_rules: list[str] = Field(default_factory=list)
    pii_findings: list[PiiFinding] = Field(default_factory=list)
    redacted: bool = False

    @property
    def pii_types(self) -> list[str]:
        return [finding.entity_type for finding in self.pii_findings]


def _normalize_for_injection_scan(text: str) -> str:
    """Normalize Unicode homoglyphs and strip zero-width obfuscation characters."""
    normalized = unicodedata.normalize("NFKC", text)
    stripped = _ZERO_WIDTH_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _is_valid_luhn(digits: str) -> bool:
    """Validate a candidate payment card number with the Luhn checksum."""
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse_digits = digits[::-1]
    for idx, ch in enumerate(reverse_digits):
        val = ord(ch) - 48
        if idx % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return total % 10 == 0


def _is_valid_tckn(digits: str) -> bool:
    """Validate an 11-digit Turkish TC Kimlik No using its official two-digit checksum."""
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    nums = [ord(ch) - 48 for ch in digits]
    odd_sum = nums[0] + nums[2] + nums[4] + nums[6] + nums[8]
    even_sum = nums[1] + nums[3] + nums[5] + nums[7]
    check_10 = ((odd_sum * 7) - even_sum) % 10
    if check_10 != nums[9]:
        return False
    check_11 = sum(nums[:10]) % 10
    return check_11 == nums[10]


def sanitize_untrusted_passage(text: str) -> str:
    """Strip zero-width obfuscation and neutralize chat role-delimiter spoofing in external text.

    Applied to untrusted web search snippets and scraped web pages before indexing or
    prompt injection so retrieved content cannot forge system/developer message boundaries.
    """
    if not text:
        return ""
    stripped = _ZERO_WIDTH_RE.sub("", text)
    return _ROLE_SPOOF_RE.sub("[filtered-delimiter]", stripped)


def detect_prompt_injection(text: str) -> list[str]:
    """Return all triggered prompt-injection / exfiltration rule IDs for ``text``."""
    if not text or not text.strip():
        return []
    cleaned = _normalize_for_injection_scan(text)
    return [rule.rule_id for rule in _INJECTION_RULES if rule.pattern.search(cleaned)]


def _redact_pii_in_text(text: str) -> tuple[str, dict[PiiEntityType, tuple[str, int]]]:
    """Detect and replace secrets/PII with deterministic placeholders."""
    counts: dict[PiiEntityType, tuple[str, int]] = {}
    current = text

    def _record(entity: PiiEntityType, placeholder: str, n: int) -> None:
        if n <= 0:
            return
        prev_placeholder, prev_count = counts.get(entity, (placeholder, 0))
        counts[entity] = (prev_placeholder, prev_count + n)

    # 1. API keys / secrets first so embedded hyphens don't interfere with other patterns
    current, n_keys = _API_KEY_RE.subn("[REDACTED_API_KEY]", current)
    _record("api_key", "[REDACTED_API_KEY]", n_keys)

    # 2. Email addresses
    current, n_emails = _EMAIL_RE.subn("[REDACTED_EMAIL]", current)
    _record("email", "[REDACTED_EMAIL]", n_emails)

    # 3. Credit card numbers (Luhn-verified)
    card_hits = 0

    def _replace_card(match: re.Match[str]) -> str:
        nonlocal card_hits
        raw = match.group(0)
        digits = re.sub(r"[^0-9]", "", raw)
        if _is_valid_luhn(digits):
            card_hits += 1
            return "[REDACTED_CARD]"
        return raw

    current = _CARD_CANDIDATE_RE.sub(_replace_card, current)
    _record("credit_card", "[REDACTED_CARD]", card_hits)

    # 4. Turkish National ID / TC Kimlik No (official checksum-verified)
    tckn_hits = 0

    def _replace_tckn(match: re.Match[str]) -> str:
        nonlocal tckn_hits
        raw = match.group(0)
        if _is_valid_tckn(raw):
            tckn_hits += 1
            return "[REDACTED_TCKN]"
        return raw

    current = _TCKN_CANDIDATE_RE.sub(_replace_tckn, current)
    _record("tr_national_id", "[REDACTED_TCKN]", tckn_hits)

    # 5. Phone numbers
    current, n_phones = _PHONE_RE.subn("[REDACTED_PHONE]", current)
    _record("phone", "[REDACTED_PHONE]", n_phones)

    return current, counts


def screen_query_input(
    query: str,
    chat_history: list[ChatMessage] | None = None,
    *,
    custom_instructions: str | None = None,
    config: InputGuardrailConfig | None = None,
) -> GuardrailDecision:
    """Screen a user query, optional chat history, and optional thread instructions.

    - Runs purely in-process at $0 API cost and sub-millisecond latency.
    - When ``config.enabled`` is False, returns the input untouched.
    - When ``config.block_prompt_injection`` is True, rejects explicit injection attempts
      in the active query, prior user messages, or thread custom instructions.
    - When ``config.pii_mode == "redact"``, replaces detected secrets/PII with safe
      placeholders before the payload reaches external LLM providers.
    - When ``config.pii_mode == "block"``, rejects requests containing secrets/PII.
    """
    cfg = config or InputGuardrailConfig()
    history = list(chat_history or [])
    cleaned_instructions = (
        custom_instructions.strip() if custom_instructions and custom_instructions.strip() else None
    )

    if not cfg.enabled:
        return GuardrailDecision(
            allowed=True,
            sanitized_query=query,
            sanitized_history=history,
            sanitized_custom_instructions=cleaned_instructions,
        )

    # 1. Prompt-injection / system-prompt exfiltration scan
    triggered_rules: list[str] = []
    if cfg.block_prompt_injection:
        triggered_rules.extend(detect_prompt_injection(query))
        if cleaned_instructions:
            for rule_id in detect_prompt_injection(cleaned_instructions):
                if rule_id not in triggered_rules:
                    triggered_rules.append(rule_id)
        for msg in history:
            if msg.role == "user":
                for rule_id in detect_prompt_injection(msg.content):
                    if rule_id not in triggered_rules:
                        triggered_rules.append(rule_id)

        if triggered_rules:
            return GuardrailDecision(
                allowed=False,
                sanitized_query=query,
                sanitized_history=history,
                sanitized_custom_instructions=cleaned_instructions,
                blocked_reason=(
                    "Request blocked by input safety guardrails: potential prompt injection "
                    f"detected ({', '.join(triggered_rules)})."
                ),
                triggered_rules=triggered_rules,
            )

    # 2. PII & secret scanning / redaction
    if cfg.pii_mode == "off":
        return GuardrailDecision(
            allowed=True,
            sanitized_query=query,
            sanitized_history=history,
            sanitized_custom_instructions=cleaned_instructions,
        )

    merged_counts: dict[PiiEntityType, tuple[str, int]] = {}

    def _merge(part: dict[PiiEntityType, tuple[str, int]]) -> None:
        for entity, (placeholder, count) in part.items():
            prev_placeholder, prev_count = merged_counts.get(entity, (placeholder, 0))
            merged_counts[entity] = (prev_placeholder, prev_count + count)

    sanitized_query, q_counts = _redact_pii_in_text(query)
    _merge(q_counts)

    sanitized_instructions: str | None = cleaned_instructions
    if cleaned_instructions is not None:
        sanitized_instructions, instr_counts = _redact_pii_in_text(cleaned_instructions)
        _merge(instr_counts)

    sanitized_history: list[ChatMessage] = []
    for msg in history:
        redacted_content, m_counts = _redact_pii_in_text(msg.content)
        _merge(m_counts)
        sanitized_history.append(msg.model_copy(update={"content": redacted_content}))

    findings = [
        PiiFinding(entity_type=entity, placeholder=placeholder, count=count)
        for entity, (placeholder, count) in merged_counts.items()
    ]

    if findings and cfg.pii_mode == "block":
        entity_list = ", ".join(f.entity_type for f in findings)
        return GuardrailDecision(
            allowed=False,
            sanitized_query=query,
            sanitized_history=history,
            sanitized_custom_instructions=cleaned_instructions,
            blocked_reason=(
                "Request blocked by input PII guardrail: remove sensitive data "
                f"before querying ({entity_list})."
            ),
            triggered_rules=[f"pii:{f.entity_type}" for f in findings],
            pii_findings=findings,
            redacted=False,
        )

    return GuardrailDecision(
        allowed=True,
        sanitized_query=sanitized_query,
        sanitized_history=sanitized_history,
        sanitized_custom_instructions=sanitized_instructions,
        pii_findings=findings,
        redacted=bool(findings),
    )

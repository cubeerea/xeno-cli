"""Secret scanning on all outbound model context (PRD S11.3).

Every prompt leaving this process passes through here first. This is a Phase 0
deliverable rather than a hardening pass because model calls begin in Phase 0 —
a scanner added in Phase 3 would have spent three phases not running.

Two detectors, because they fail differently:

  * KNOWN PREFIXES catch real credentials with near-zero false positives, but
    only ones whose format someone thought to encode here.
  * HIGH ENTROPY catches the rest, at the cost of occasionally redacting a
    legitimate digest.

Redaction is deliberately biased toward over-redacting. A redacted git SHA
costs the model a little context; a leaked deploy key costs considerably more.
The known exception is pure-hex digests and UUIDs, which are excluded by
default — they are pervasive in real repositories (lockfiles, git history,
test fixtures) and redacting them wholesale would degrade every codebase map
the harness ever builds.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

#: (label, compiled pattern). Ordered; first match wins.
KNOWN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openrouter_api_key", re.compile(r"sk-or-v1-[A-Za-z0-9]{32,}")),
    ("openai_api_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}")),
    ("github_fine_grained_pat", re.compile(r"github_pat_[A-Za-z0-9_]{50,}")),
    ("gitlab_token", re.compile(r"glpat-[A-Za-z0-9_\-]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("digitalocean_token", re.compile(r"\bdop_v1_[a-f0-9]{64}\b")),
    ("sendgrid_api_key", re.compile(r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b")),
    ("stripe_secret_key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{24,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
)

#: Assignments to a secret-shaped name. Catches credentials with no
#: distinguishing prefix, which the entropy detector would otherwise have to
#: guess at (a 20-char lowercase password has unremarkable entropy).
ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    \b(?P<name>[A-Za-z0-9_\-\.]*
        (?:pass(?:wd|word)?|secret|token|api[_\-]?key|access[_\-]?key
          |client[_\-]?secret|auth|credential|private[_\-]?key|passphrase)
     [A-Za-z0-9_\-\.]*)
    \s*[:=]\s*
    (?P<quote>["']?)
    (?P<value>[^\s"',;]{8,200})
    (?P=quote)
    """
)

#: Candidate tokens for the entropy detector.
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")

_PURE_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: Values that look secret-ish but are placeholders. Redacting these produces
#: noisy findings and teaches users to ignore the log.
_PLACEHOLDERS = frozenset(
    {
        "changeme", "your_api_key_here", "xxxxxxxxxxxx", "none", "null", "todo",
        "example", "placeholder", "redacted", "dummy", "test", "fake",
    }
)


@dataclass(frozen=True, slots=True)
class Finding:
    """One detected secret. Never carries the secret itself."""

    label: str
    detector: str  # "known_prefix" | "assignment" | "high_entropy"
    preview: str  # first 4 chars + length only, safe to log


@dataclass(frozen=True, slots=True)
class ScanResult:
    text: str  # redacted; safe to send
    findings: tuple[Finding, ...]

    @property
    def clean(self) -> bool:
        return not self.findings


def shannon_entropy(value: str) -> float:
    """Bits per character. ~4.0+ indicates a random-looking token; English
    prose sits near 2.5 and identifiers lower still."""
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


#: Bound on the per-scanner memo (see `SecretScanner.scan`). Sized to cover a
#: prompt's static blocks plus a long accumulated history without growing with
#: the run — the entries are prompt-sized strings, so this is a memory ceiling
#: as much as a cache policy.
_SCAN_CACHE_ENTRIES = 64


class SecretScanner:
    """Redacts credentials from outbound context.

    Stateless across calls apart from configuration, so it is safe to share one
    instance for a whole run.
    """

    def __init__(
        self,
        *,
        entropy_threshold: float = 4.0,
        min_entropy_length: int = 24,
        redact_hex_digests: bool = False,
        extra_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (),
    ) -> None:
        self.entropy_threshold = entropy_threshold
        self.min_entropy_length = min_entropy_length
        self.redact_hex_digests = redact_hex_digests
        self.patterns = (*KNOWN_PATTERNS, *extra_patterns)
        self._cache: dict[str, ScanResult] = {}

    def scan(self, text: str) -> ScanResult:
        """Return `text` with every detected secret replaced by a stable marker.

        Markers are indexed rather than hashed. A hash of the secret would be a
        weak leak of the very thing being redacted, and it would also travel
        into the prompt cache.

        Results are memoized because `xeno.security.outbound.sanitize` scans
        EVERY layer of EVERY prompt, and most of those layers are byte-identical
        call after call: the SYSTEM block is fingerprinted as constant per node
        for the process lifetime (PRD T8), and each accumulated history turn is
        immutable once appended, so an N-turn conversation otherwise re-scans
        the same N-1 turns on every single call. Safe to cache because the scan
        depends on nothing but `text` and this instance's configuration, none of
        which is mutable after construction.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        result = self._scan_uncached(text)
        if len(self._cache) >= _SCAN_CACHE_ENTRIES:
            # Plain FIFO eviction on a dict (insertion-ordered since 3.7).
            # A true LRU would buy little here: the hot entries are the system
            # block and the history tail, and both are re-inserted as soon as
            # they are evicted.
            del self._cache[next(iter(self._cache))]
        self._cache[text] = result
        return result

    def _scan_uncached(self, text: str) -> ScanResult:
        spans: list[tuple[int, int, str, str]] = []  # start, end, label, detector

        for label, pattern in self.patterns:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), label, "known_prefix"))

        for match in ASSIGNMENT_RE.finditer(text):
            value = match.group("value")
            if self._is_placeholder(value):
                continue
            spans.append(
                (
                    match.start("value"),
                    match.end("value"),
                    match.group("name").lower(),
                    "assignment",
                )
            )

        for match in _TOKEN_RE.finditer(text):
            token = match.group(0)
            if not self._is_high_entropy(token):
                continue
            spans.append((match.start(), match.end(), "high_entropy_string", "high_entropy"))

        return self._redact(text, spans)

    def _is_placeholder(self, value: str) -> bool:
        return value.strip().strip("<>\"'").lower() in _PLACEHOLDERS

    def _is_high_entropy(self, token: str) -> bool:
        if len(token) < self.min_entropy_length:
            return False
        if self._is_placeholder(token):
            return False
        if _PURE_HEX_RE.match(token) or _UUID_RE.match(token):
            # Hex draws from 16 symbols, so its entropy CANNOT exceed 4.0
            # bits/char and in practice sits below it. Deferring to the entropy
            # threshold here would make `redact_hex_digests` do nothing at all,
            # so the decision is made on the flag and the length instead.
            #
            # Default False: git SHAs, content digests, and lockfile hashes are
            # pervasive in real repositories and are not credentials. See the
            # module docstring.
            return self.redact_hex_digests
        return shannon_entropy(token) >= self.entropy_threshold

    def _redact(
        self, text: str, spans: list[tuple[int, int, str, str]]
    ) -> ScanResult:
        if not spans:
            return ScanResult(text=text, findings=())

        # Longest-match-first at each start, then drop anything overlapping an
        # already-accepted span, so a private key block is redacted whole
        # rather than shredded by the entropy detector matching inside it.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        accepted: list[tuple[int, int, str, str]] = []
        cursor = -1
        for start, end, label, detector in spans:
            if start < cursor:
                continue
            accepted.append((start, end, label, detector))
            cursor = end

        findings: list[Finding] = []
        out: list[str] = []
        last = 0
        for index, (start, end, label, detector) in enumerate(accepted, start=1):
            secret = text[start:end]
            findings.append(
                Finding(
                    label=label,
                    detector=detector,
                    preview=f"{secret[:4]}…({len(secret)} chars)",
                )
            )
            out.append(text[last:start])
            out.append(f"[REDACTED:{label}:#{index}]")
            last = end
        out.append(text[last:])
        return ScanResult(text="".join(out), findings=tuple(findings))

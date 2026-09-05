"""PERCEPTION LAYER - the sensory cortex.

Purpose
    Ingest text, voice, images and structured data and normalise each into a
    shared internal representation (``Percept``) that every other module can
    consume without caring where it came from.

Technique
    * Text: light NLP - intent keywords, entity extraction, sentiment/urgency
      lexicons, embedding.
    * Voice: speech-to-text adapter (Whisper / cloud STT) -> text pipeline,
      carrying the STT confidence forward.
    * Image: vision-caption / OCR adapter -> text pipeline; caption becomes the
      canonical text, raw bytes are *not* stored in memory.
    * Structured data: schema-aware flattening into a textual description plus
      the original object in ``metadata``.

Communication
    ``perceive()`` returns a ``Percept`` and publishes ``perception.percept_ready``.

Data
    Reads nothing persistent. Writes nothing persistent (memory decides what
    to keep).

Improvement over time
    The intent/entity lexicons are part of the prompt/config library and can be
    extended by approved upgrade proposals; adapters can be swapped for better
    models without touching downstream code.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from nexus_brain.core.bus import EventBus, Topics
from nexus_brain.core.schemas import Modality, Percept
from nexus_brain.perception.embedder import Embedder, HashingEmbedder, tokenize

# Simple lexicons - deliberately explainable. A production system would use an
# LLM/NLU model here; the *interface* stays the same.
INTENT_PATTERNS: dict[str, list[str]] = {
    "schedule": ["schedule", "set up a meeting", "arrange a meeting", "book a", "book an", "put on my calendar", "add to my calendar", "remind me"],
    "email": ["email", "mail", "send a message", "write to"],
    "search": ["search", "find", "look up", "lookup", "what is", "who is", "google"],
    "summarize": ["summarize", "summarise", "summary", "tl;dr", "recap"],
    "calculate": ["calculate", "compute", "how much", "total", "sum", "multiply", "%"],
    "remember": ["remember that", "remember my", "remember i", "please remember", "note that", "keep in mind", "my name is", "i prefer", "i like", "call me"],
    "recall": ["what do you remember", "what do you know about", "what did i", "do you recall", "last time", "remind me what"],
    "greeting": ["__greeting__"],  # handled by regex, see detect_intents
    "feedback_positive": ["thanks", "thank you", "great", "perfect", "awesome", "well done"],
    "feedback_negative": ["wrong", "that's not", "useless", "bad answer", "no,", "incorrect"],
    "payment": ["pay", "payment", "transfer money", "wire", "purchase", "buy"],
    "delete": ["delete", "remove", "erase", "wipe"],
    "shell": ["shell command", "run a command", "bash", "terminal", "sudo"],
    "weather": ["weather", "forecast", "temperature", "rain"],
    "help": ["help", "how do i", "how to", "explain"],
    "explain_self": ["are you conscious", "do you feel", "are you sentient", "how do you work"],
}

_POSITIVE = {"great", "thanks", "thank", "love", "awesome", "perfect", "good", "nice", "happy", "excellent"}
_NEGATIVE = {"wrong", "bad", "hate", "angry", "frustrated", "useless", "terrible", "annoyed", "broken", "again"}
_URGENT = {"urgent", "asap", "immediately", "now", "right away", "emergency", "deadline", "today", "hurry"}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_DATE_RE = re.compile(
    r"\b(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}(:\d{2})?\s?(am|pm)|\d{4}-\d{2}-\d{2}|next week|this week)\b",
    re.I,
)
_NAME_RE = re.compile(r"\b(?:my name is|call me) ([A-Z][a-z]+)", re.I)
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good (?:morning|afternoon|evening))\b", re.I)
_MONEY_RE = re.compile(r"[$€£]\s?\d[\d,]*(\.\d+)?|\b\d+\s?(dollars|usd|eur|euros|pounds)\b", re.I)


class SpeechToText:
    """Adapter interface for voice. Default just passes a provided transcript."""

    def transcribe(self, audio: Any) -> tuple[str, float]:
        if isinstance(audio, dict) and "transcript" in audio:
            return str(audio["transcript"]), float(audio.get("confidence", 0.85))
        if isinstance(audio, str):
            return audio, 0.85
        raise ValueError("No STT backend configured; pass {'transcript': ..., 'confidence': ...}")


class ImageCaptioner:
    """Adapter interface for images. Default uses provided caption / OCR text."""

    def describe(self, image: Any) -> tuple[str, float]:
        if isinstance(image, dict):
            parts = [str(image.get("caption", "")), str(image.get("ocr_text", ""))]
            text = " ".join(p for p in parts if p).strip()
            if text:
                return text, float(image.get("confidence", 0.8))
        raise ValueError("No vision backend configured; pass {'caption': ..., 'ocr_text': ...}")


class Perception:
    def __init__(
        self,
        bus: EventBus,
        embedder: Optional[Embedder] = None,
        stt: Optional[SpeechToText] = None,
        captioner: Optional[ImageCaptioner] = None,
        intent_patterns: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.bus = bus
        self.embedder = embedder or HashingEmbedder()
        self.stt = stt or SpeechToText()
        self.captioner = captioner or ImageCaptioner()
        self.intent_patterns = intent_patterns or INTENT_PATTERNS

    # ------------------------------------------------------------------ #
    def perceive(
        self,
        payload: Any,
        modality: Modality | str = Modality.TEXT,
        session_id: Optional[str] = None,
        source: str = "user",
    ) -> Percept:
        modality = Modality(modality)
        confidence = 1.0
        metadata: dict[str, Any] = {}

        if modality == Modality.TEXT:
            text = str(payload)
        elif modality == Modality.VOICE:
            text, confidence = self.stt.transcribe(payload)
            metadata["stt_confidence"] = confidence
        elif modality == Modality.IMAGE:
            text, confidence = self.captioner.describe(payload)
            text = f"[image] {text}"
            metadata["vision_confidence"] = confidence
        else:  # STRUCTURED
            text = self._flatten_structured(payload)
            metadata["structured"] = payload

        percept = Percept(
            modality=modality,
            text=text,
            embedding=self.embedder.embed(text),
            entities=self.extract_entities(text),
            intents=self.detect_intents(text),
            sentiment=self.sentiment(text),
            urgency=self.urgency(text),
            confidence=confidence,
            source=source,
            session_id=session_id,
            metadata=metadata,
        )
        self.bus.publish(
            Topics.PERCEPT_READY,
            {"percept_id": percept.id, "modality": modality.value, "intents": percept.intents},
            source="perception",
        )
        return percept

    # ------------------------------------------------------------------ #
    def detect_intents(self, text: str) -> list[str]:
        lowered = f" {text.lower()} "
        found = [
            intent
            for intent, pats in self.intent_patterns.items()
            if any(p in lowered for p in pats)
        ]
        if _GREETING_RE.match(text):
            found.append("greeting")
        if "recall" in found and "remember" in found:
            found.remove("remember")  # "what do you remember" is a question, not a request to store
        if "explain_self" in found:
            found = [f for f in found if f not in ("help", "chat")]
        return found or ["chat"]

    def extract_entities(self, text: str) -> list[str]:
        ents: list[str] = []
        ents += [f"email:{m}" for m in _EMAIL_RE.findall(text)]
        ents += [f"date:{m[0]}" for m in _DATE_RE.findall(text)]
        ents += [f"person:{m}" for m in _NAME_RE.findall(text)]
        ents += [f"money:{m.group(0).strip()}" for m in _MONEY_RE.finditer(text)]
        return ents

    @staticmethod
    def sentiment(text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 0.0
        pos = sum(1 for t in toks if t in _POSITIVE)
        neg = sum(1 for t in toks if t in _NEGATIVE)
        if "!" in text and neg:
            neg += 1
        if text.isupper() and len(text) > 8:
            neg += 1
        if pos == neg == 0:
            return 0.0
        return max(-1.0, min(1.0, (pos - neg) / (pos + neg)))

    @staticmethod
    def urgency(text: str) -> float:
        lowered = text.lower()
        hits = sum(1 for u in _URGENT if u in lowered)
        score = min(1.0, 0.35 * hits)
        if "!" in text:
            score = min(1.0, score + 0.15)
        return score

    @staticmethod
    def _flatten_structured(payload: Any) -> str:
        if isinstance(payload, dict):
            parts = [f"{k}: {v}" for k, v in payload.items()]
            return "[structured] " + "; ".join(parts)
        return "[structured] " + json.dumps(payload, default=str)

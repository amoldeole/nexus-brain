from nexus_brain.core.schemas import Modality
from nexus_brain.perception.embedder import HashingEmbedder, cosine
from nexus_brain.perception.perception import Perception


def test_text_percept_has_intents_entities_and_embedding(bus):
    p = Perception(bus)
    pct = p.perceive("Please schedule a meeting with dana@example.com tomorrow, it's urgent!")
    assert pct.modality == Modality.TEXT
    assert "schedule" in pct.intents
    assert "email:dana@example.com" in pct.entities
    assert "date:tomorrow" in pct.entities
    assert pct.urgency > 0.3
    assert pct.embedding and len(pct.embedding) == p.embedder.dim
    assert bus.history[-1].topic == "perception.percept_ready"


def test_voice_and_image_and_structured_modalities_normalise_to_text(bus):
    p = Perception(bus)
    voice = p.perceive({"transcript": "what's the weather in Paris", "confidence": 0.6}, modality="voice")
    assert voice.modality == Modality.VOICE and "weather" in voice.intents and voice.confidence == 0.6
    image = p.perceive({"caption": "a receipt", "ocr_text": "total $42.10"}, modality="image")
    assert image.text.startswith("[image]") and "money:$42.10" in image.entities
    structured = p.perceive({"ticket": 123, "priority": "high"}, modality="structured")
    assert "ticket: 123" in structured.text and structured.metadata["structured"]["priority"] == "high"


def test_sentiment_lexicon():
    assert Perception.sentiment("thanks, that was great") > 0
    assert Perception.sentiment("this is wrong and useless") < 0
    assert Perception.sentiment("schedule a meeting") == 0


def test_greeting_and_recall_disambiguation(bus):
    p = Perception(bus)
    assert "greeting" in p.perceive("Hello there").intents
    assert "greeting" not in p.perceive("the highest mountain").intents  # 'hi' inside a word
    intents = p.perceive("what do you remember about me?").intents
    assert "recall" in intents and "remember" not in intents


def test_hashing_embedder_is_deterministic_and_similarity_sensible():
    e = HashingEmbedder(dim=256)
    a, b = e.embed("schedule a meeting with Dana"), e.embed("schedule a meeting with Dana")
    assert a == b
    sim_close = cosine(e.embed("book a meeting with Dana tomorrow"), e.embed("schedule a meeting with Dana"))
    sim_far = cosine(e.embed("the weather in Paris is sunny"), e.embed("schedule a meeting with Dana"))
    assert sim_close > sim_far

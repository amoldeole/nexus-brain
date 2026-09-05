from datetime import timedelta

from nexus_brain.core.config import RetrievalWeights, WorkingMemoryConfig
from nexus_brain.core.schemas import ActionSpec, MemoryKind, Skill, utcnow
from nexus_brain.memory.long_term import LongTermMemory
from nexus_brain.memory.store import InMemoryVectorStore
from nexus_brain.memory.working import WorkingMemory


def test_retrieval_scores_relevance_recency_importance(bus):
    ltm = LongTermMemory(bus, weights=RetrievalWeights(relevance=0.6, recency=0.2, importance=0.2))
    ltm.remember_fact("User prefers morning meetings", importance=0.9)
    ltm.remember_fact("User's favourite colour is blue", importance=0.3)
    ltm.remember_episode("User asked about the weather in Paris", importance=0.5)
    hits = ltm.retrieve("when does the user like to have meetings?", k=3)
    assert hits and hits[0].item.content == "User prefers morning meetings"
    assert hits[0].score >= hits[-1].score
    assert hits[0].item.access_count == 1  # retrieval strengthens the trace


def test_recency_decay_prefers_newer_memory_when_equally_relevant(bus):
    ltm = LongTermMemory(bus, weights=RetrievalWeights(relevance=0.4, recency=0.5, importance=0.1, recency_half_life_hours=24))
    old = ltm.remember_episode("meeting with Dana about the launch", importance=0.5)
    old.created_at = utcnow() - timedelta(days=10)
    ltm.store.update(old)
    ltm.remember_episode("meeting with Dana about the launch", importance=0.5)
    hits = ltm.retrieve("meeting with Dana", k=2)
    assert hits[0].recency > hits[1].recency


def test_semantic_dedupe_and_decay_and_forget(bus):
    ltm = LongTermMemory(bus)
    a = ltm.remember_fact("User's name is Priya")
    b = ltm.remember_fact("User's name is Priya")
    assert a.id == b.id and ltm.stats()["semantic"] == 1
    weak = ltm.remember_episode("trivial remark", importance=0.05)
    weak.created_at = utcnow() - timedelta(days=30)
    weak.last_accessed = weak.created_at
    ltm.store.update(weak)
    assert ltm.decay() >= 1
    assert ltm.forget() == 1


def test_procedural_skill_matching_and_stats(bus):
    ltm = LongTermMemory(bus)
    ltm.add_skill(Skill(name="s", description="d", triggers=["schedule"], steps=[ActionSpec("check_calendar", {})]))
    found = ltm.find_skills("please schedule something", ["schedule"])
    assert found and found[0][0].name == "s"
    ltm.record_skill_outcome("s", True)
    ltm.record_skill_outcome("s", False)
    assert ltm.skills["s"].attempts == 2 and 0 < ltm.skills["s"].success_rate < 1


def test_persistence_roundtrip(tmp_path, bus):
    store = InMemoryVectorStore(tmp_path / "mem.json")
    ltm = LongTermMemory(bus, store=store, skills_path=tmp_path / "skills.json")
    ltm.remember_fact("User lives in Lisbon")
    ltm.add_skill(Skill(name="k", description="d", triggers=["x"], steps=[]))
    reloaded = LongTermMemory(bus, store=InMemoryVectorStore(tmp_path / "mem.json"), skills_path=tmp_path / "skills.json")
    assert any("Lisbon" in i.content for i in reloaded.store.all([MemoryKind.SEMANTIC]))
    assert "k" in reloaded.skills


def test_working_memory_keeps_recent_summarises_old_and_consolidates(bus):
    consolidated = []
    wm = WorkingMemory(bus, WorkingMemoryConfig(token_budget=60, keep_last_turns=2), consolidate=consolidated.append)
    wm.add_turn("user", "ok")  # low salience -> discarded first
    for i in range(6):
        wm.add_turn("user", f"Remember the deadline for project {i} is friday and the budget is important")
        wm.add_turn("assistant", f"Noted project {i}.")
    assert len(wm.turns) <= 4
    assert wm.used_tokens <= 60 + 20  # protected turns may slightly exceed
    assert wm.summary  # older turns were summarised, not lost
    assert wm.compressions > 0
    assert consolidated and all(t.salience >= 0.6 for t in consolidated)
    assert "[earlier conversation summary]" in wm.context_window()

from __future__ import annotations

from typing import Any

from .scribe_diagnostics import ScribeToken


AGENT_PHRASES = (
    "i'm erica",
    "i am erica",
    "how can i help",
    "toyota of",
    "lexington toyota",
    "what type of service",
    "just to confirm",
    "dealership is closed",
    "did you want to transfer",
    "transferring you",
    "i can help with that",
)


def _speaker_text(words: list[ScribeToken]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}

    for word in words:
        speaker = (word.speaker_id or "").strip()
        text = word.text.strip()

        if speaker and text:
            grouped.setdefault(speaker, []).append(text)

    return {
        speaker: " ".join(parts)
        for speaker, parts in grouped.items()
    }


def _speaker_turns(
    words: list[ScribeToken],
    speaker_id: str,
) -> list[dict[str, Any]]:
    timed: list[tuple[float, float, str]] = []

    for word in words:
        if word.speaker_id != speaker_id:
            continue

        if word.start is None or word.end is None:
            continue

        text = word.text.strip()
        if not text:
            continue

        timed.append(
            (
                float(word.start),
                float(word.end),
                text,
            )
        )

    timed.sort(key=lambda item: item[0])

    turns: list[dict[str, Any]] = []

    for start, end, text in timed:
        if (
            turns
            and start - float(turns[-1]["end"]) <= 0.8
        ):
            turns[-1]["end"] = max(
                float(turns[-1]["end"]),
                end,
            )
            turns[-1]["text"] += f" {text}"
        else:
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    return turns


def _speaker_diagnostics(
    words: list[ScribeToken],
    speaker_id: str,
) -> dict[str, Any]:
    timed = [
        word
        for word in words
        if word.speaker_id == speaker_id
        and word.start is not None
        and word.end is not None
        and word.text.strip()
    ]

    timed.sort(key=lambda word: float(word.start or 0.0))

    if not timed:
        raise RuntimeError(
            f"Speaker {speaker_id} has no timed spoken words."
        )

    return {
        "speaker_id": speaker_id,
        "first_word_start": float(timed[0].start),
        "last_word_end": float(timed[-1].end),
        "spoken_word_count": len(timed),
        "total_speech_duration": round(
            sum(
                max(
                    0.0,
                    float(word.end) - float(word.start),
                )
                for word in timed
            ),
            3,
        ),
        "transcript": " ".join(
            word.text.strip()
            for word in timed
        ),
    }


def infer_customer_speaker(
    words: list[ScribeToken],
) -> tuple[str, dict[str, Any]]:
    texts = _speaker_text(words)

    if len(texts) < 2:
        raise RuntimeError(
            "Need at least two diarized speakers to isolate the customer."
        )

    phrase_hits: dict[str, list[str]] = {}
    scores: dict[str, int] = {}

    for speaker, text in texts.items():
        lowered = text.casefold()
        hits = [
            phrase
            for phrase in AGENT_PHRASES
            if phrase in lowered
        ]
        phrase_hits[speaker] = hits
        scores[speaker] = len(hits)

    agent = max(scores, key=scores.get)

    if scores[agent] <= 0:
        raise RuntimeError(
            "Could not infer dealership-agent speaker from role phrases."
        )

    agent_turns = _speaker_turns(words, agent)

    if not agent_turns:
        raise RuntimeError(
            "Dealership-agent speaker has no timed greeting turn."
        )

    greeting = next(
        (
            turn
            for turn in agent_turns
            if any(
                phrase in str(turn["text"]).casefold()
                for phrase in AGENT_PHRASES
            )
        ),
        agent_turns[0],
    )

    greeting_end = float(greeting["end"])

    candidates: list[dict[str, Any]] = []

    for speaker in texts:
        if speaker == agent:
            continue

        diagnostics = _speaker_diagnostics(
            words,
            speaker,
        )

        response_turns = [
            turn
            for turn in _speaker_turns(words, speaker)
            if float(turn["start"]) >= greeting_end
        ]

        diagnostics["first_post_greeting_turn"] = (
            response_turns[0]
            if response_turns
            else None
        )

        candidates.append(diagnostics)

    eligible = [
        candidate
        for candidate in candidates
        if candidate["first_post_greeting_turn"] is not None
    ]

    if not eligible:
        raise RuntimeError(
            "No non-agent speaker responded after the dealership greeting."
        )

    selected = min(
        eligible,
        key=lambda candidate: (
            float(
                candidate[
                    "first_post_greeting_turn"
                ]["start"]
            ),
            -float(candidate["total_speech_duration"]),
            str(candidate["speaker_id"]),
        ),
    )

    customer = str(selected["speaker_id"])

    return customer, {
        "agent_speaker": agent,
        "customer_speaker": customer,
        "agent_greeting_turn": greeting,
        "agent_phrase_scores": scores,
        "agent_phrase_hits": phrase_hits,
        "candidate_speakers": candidates,
        "customer_selection_reason": (
            f"Selected {customer}: earliest non-agent spoken turn "
            f"after dealership greeting ended at {greeting_end:.3f}s."
        ),
    }


def build_customer_segments(
    words: list[ScribeToken],
    customer_speaker: str,
) -> list[dict[str, Any]]:
    timed: list[tuple[float, float, str]] = []

    for word in words:
        if word.speaker_id != customer_speaker:
            continue

        if word.start is None or word.end is None:
            continue

        text = word.text.strip()
        if not text:
            continue

        timed.append(
            (
                float(word.start),
                float(word.end),
                text,
            )
        )

    timed.sort(key=lambda item: item[0])

    if not timed:
        raise RuntimeError(
            "No timed customer words were available."
        )

    segments: list[dict[str, Any]] = []

    current_start, current_end, first_text = timed[0]
    current_words = [first_text]

    for start, end, text in timed[1:]:
        gap = start - current_end
        projected_duration = end - current_start

        if gap <= 0.8 and projected_duration <= 9.0:
            current_end = max(current_end, end)
            current_words.append(text)
        else:
            segments.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_words),
                }
            )

            current_start = start
            current_end = end
            current_words = [text]

    segments.append(
        {
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_words),
        }
    )

    return segments

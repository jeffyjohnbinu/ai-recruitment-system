"""
main.py
--------
Demonstration entry point that wires the pipeline together:

    resume file -> parsers -> ats_engine -> screening_ai -> scoring
    audio file -> speech_to_text -> ats_engine -> screening_ai -> scoring

Run with:
    python main.py --resume data/raw/sample_resume.txt --job data/raw/sample_job.txt
    python main.py --audio data/raw/interview.wav --job data/raw/sample_job.txt

For real PDF/DOCX resumes, parsers.resume_parser.parse_resume() is used
instead of the plain-text shortcut below.

For audio transcription, the speech_to_text pipeline processes voice input
through Whisper (STT) + normalization, producing clean text that feeds
into the same ATS + screening pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ats_engine.matcher import match_resume_to_job
from scoring.scorer import compute_final_score
from utils.logger import get_logger

logger = get_logger("main")


def run_resume_pipeline(resume_path: Path, job_path: Path) -> None:
    """Run the document-based pipeline: resume -> ATS -> screening -> score."""
    logger.info("Starting resume pipeline: resume=%s job=%s", resume_path, job_path)

    resume_text = resume_path.read_text(encoding="utf-8")
    job_text = job_path.read_text(encoding="utf-8")

    match_result = match_resume_to_job(resume_text, job_text)

    ai_recommendation = "advance" if match_result.is_shortlisted else "hold"
    final = compute_final_score(match_result.score, ai_recommendation)

    print("\n--- Resume Pipeline Result ---")
    print(f"ATS match score : {match_result.score:.2f}")
    print(f"Matched keywords: {', '.join(match_result.matched_keywords) or '(none)'}")
    print(f"AI recommendation: {ai_recommendation}")
    print(f"Final score      : {final.final_score:.2f}")


def run_voice_pipeline(
    audio_path: Path,
    job_path: Path,
    accent: str | None = None,
) -> None:
    """Run the voice-based pipeline: audio -> STT -> normalize -> ATS -> score."""
    logger.info(
        "Starting voice pipeline: audio=%s job=%s accent=%s",
        audio_path,
        job_path,
        accent,
    )

    from speech_to_text import TranscriptProcessor

    processor = TranscriptProcessor(
        stt_model="whisper-1",
        language="en",
    )

    try:
        result = processor.process_audio(audio_path, accent=accent)
    except FileNotFoundError as exc:
        logger.error("Audio file not found: %s", audio_path)
        raise SystemExit(f"Error: Audio file not found: {audio_path}") from exc
    except ValueError as exc:
        logger.error("Unsupported audio format: %s", audio_path)
        raise SystemExit(f"Error: {exc}") from exc

    if not result.cleaned_text.strip():
        logger.warning("Audio transcription produced empty text")
        print("\n--- Voice Pipeline Result ---")
        print("WARNING: No speech detected in audio file.")
        return

    logger.info(
        "Transcribed %.1fs of audio -> %d chars cleaned text (interrupted=%s)",
        result.duration_seconds,
        len(result.cleaned_text),
        result.interrupted,
    )

    job_text = job_path.read_text(encoding="utf-8")

    match_result = match_resume_to_job(result.cleaned_text, job_text)

    ai_recommendation = "advance" if match_result.is_shortlisted else "hold"
    final = compute_final_score(match_result.score, ai_recommendation)

    print("\n--- Voice Pipeline Result ---")
    print(f"Audio duration  : {result.duration_seconds:.1f}s")
    print(f"Transcribed text : {result.cleaned_text[:100]}...")
    print(f"Filler words rm  : {len(result.fillers_removed)}")
    print(f"Interrupted      : {result.interrupted}")
    print(f"Silence gaps     : {len(result.silence_gaps)}")
    print(f"Language         : {result.language}")
    print(f"ATS match score  : {match_result.score:.2f}")
    print(f"Matched keywords : {', '.join(match_result.matched_keywords) or '(none)'}")
    print(f"AI recommendation: {ai_recommendation}")
    print(f"Final score      : {final.final_score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the AI recruitment pipeline on one candidate "
        "(resume or voice interview)."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to a plain-text resume file.",
    )
    parser.add_argument(
        "--job",
        type=Path,
        required=True,
        help="Path to a plain-text job description file.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Path to a voice interview audio file (wav/mp3/m4a/flac/ogg).",
    )
    parser.add_argument(
        "--accent",
        help="Speaker accent hint for Whisper (e.g., indian, british, american).",
    )

    args = parser.parse_args()

    if args.resume and args.audio:
        raise SystemExit("Error: specify --resume OR --audio, not both.")

    if not args.resume and not args.audio:
        raise SystemExit("Error: specify --resume or --audio.")

    if args.resume:
        run_resume_pipeline(args.resume, args.job)
    elif args.audio:
        run_voice_pipeline(args.audio, args.job, accent=args.accent)


if __name__ == "__main__":
    main()

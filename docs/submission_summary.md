# ClassLens AI — Submission Summary

*(Maximum 300 words. Replace all **[PLACEHOLDER]** markers with real results before submitting.)*

---

## Approach

ClassLens AI processes an online class recording through a seven-step pipeline: validation, audio extraction, speech transcription, audio feature analysis, video frame sampling, AI evaluation, and Python-calculated scoring.

Recordings are transcribed with OpenAI Whisper, producing timestamped segments used as primary evidence. Audio features (silence ratio, signal-to-noise estimate, clipping) are computed locally with NumPy — no API call required. Up to twelve frames are sampled with OpenCV and described with GPT-4o vision. All evidence is bundled into a typed `EvidencePackage` and sent to GPT-4o with a structured system prompt that specifies a five-dimension rubric: Teaching Clarity (30%), Engagement (25%), Lesson Structure (20%), Audio Intelligibility (15%), and Visual Support (10%). The model returns dimension scores with evidence citations; Python then independently calculates the weighted score and maps it to a rating using configurable thresholds.

## Why This Approach

Using Whisper for transcription and GPT-4o for reasoning keeps the pipeline lean without requiring any model training. Separating evidence extraction from AI reasoning means the scoring engine can validate and override the model's output — the LLM cannot fabricate a rating. Pydantic enforces schema integrity at every stage.

## Testing Results

**[PLACEHOLDER — replace with actual results after testing on 8x8 sample recordings]**
- All 128 automated unit tests pass (no API key required).
- Application confirmed running locally: `streamlit run app.py`.
- Tested on sample recordings: **[describe actual results here]**.
- Hosted URL: **[insert after deployment]**.

## Limitations

Transcript quality degrades with noisy audio, affecting three of five dimensions. Frame coverage is limited to twelve samples. The rubric weights are reasoned defaults not calibrated against labelled data. Very long recordings have transcripts truncated at approximately 6 000 words.

---

*Word count: ~250 words (excluding placeholders)*

# ClassLens AI — Evaluation Methodology

## The Problem

An online class recording must be classified as Poor, Average, Good, or Excellent based on the quality of teaching.  The challenge does not prescribe a rubric, so the methodology below is a reasoned hypothesis.  The weights and thresholds are provisional and should be recalibrated if reference-labelled data becomes available.

---

## What Can Be Observed

From a typical online class recording, the following evidence is extractable without human review:

| Evidence type | Source | Reliability |
|--------------|--------|-------------|
| Teacher speech text | Whisper transcription | High for clear audio; degrades with noise or accent |
| Speech timestamps | Whisper segment timestamps | Generally accurate to ±1 s |
| Silence patterns | Local NumPy audio analysis | Reliable |
| Signal-to-noise ratio | Local NumPy audio analysis | Approximate |
| Audio clipping | Local NumPy audio analysis | Reliable |
| On-screen content (slides, board) | GPT-4o vision on sampled frames | Limited to sampled frames only |

What cannot be reliably observed without additional sensors or data:

- Student comprehension or learning outcomes
- Student engagement beyond observable on-screen responses
- Teacher intent or lesson preparation quality
- Off-camera activity

The methodology is designed to stay within the boundaries of what the available evidence can actually support.

---

## Five Evaluation Dimensions

### Dimension 1 — Teaching Clarity (weight: 30%)

**Why 30%?**  
Clear explanation is the core purpose of a teaching session.  It has the highest individual impact on whether students can follow the lesson.

**What is assessed:**
- Concepts explained with step-by-step or structured reasoning
- Use of examples that relate to the concept being taught
- Logical progression from one idea to the next
- Avoidance of contradictory or confusing statements
- Introduction of new vocabulary or concepts with sufficient context

**Important distinctions:**
- A difficult subject does not imply poor teaching
- Speed of delivery is not scored — only clarity of explanation
- Technical errors in the subject matter are not penalised unless they confuse students

**Primary evidence source:** transcript

---

### Dimension 2 — Engagement and Checks for Understanding (weight: 25%)

**Why 25%?**  
Interactive teaching is strongly associated with better learning, but it is the second priority after basic clarity.

**What is assessed:**
- Questions posed to students (open or closed)
- Explicit invitations for students to respond or reflect
- Wait time or pause after a question (observable as silence following a question)
- Evidence of adapting explanation after a confusing moment (if audible)
- Responses to student questions when audible

**Important distinctions:**
- Absence of audible student responses does not imply poor engagement — students may be in a silent breakout or chat-based session
- High energy or entertainment value is not scored as engagement
- A single thoughtful question can score higher than many superficial ones

**Primary evidence source:** transcript (questions), audio features (pause patterns)

---

### Dimension 3 — Lesson Structure (weight: 20%)

**Why 20%?**  
Structure helps students follow a lesson and remember its content, but a well-structured poorly-explained lesson is still unhelpful.

**What is assessed:**
- Introduction of the topic and/or learning objective near the start
- Coherent progression from introduction through content to conclusion
- Understandable transitions between topics
- Examples and explanations that build on earlier material
- A recap, summary, or closing when appropriate to the duration

**Important distinctions:**
- Short recordings should not be penalised for lacking features that would not fit their duration (e.g. a 5-minute demo need not have a formal recap)
- An informal structure is not automatically poor structure

**Primary evidence source:** transcript (temporal analysis of content)

---

### Dimension 4 — Audio Intelligibility (weight: 15%)

**Why 15%?**  
Audio quality directly affects usability.  However, it is often outside the teacher's control and should not dominate the overall rating.

**What is assessed:**
- Whether speech is understandable (proxy: Whisper transcription confidence and completeness)
- Background noise level (proxy: estimated SNR)
- Clipping or distortion
- Prolonged interruptions to audio

**Important distinctions:**
- Accent is not penalised
- Loudness is not rewarded
- A deliberately quiet delivery is not poor audio

**Primary evidence source:** audio features (SNR, clipping detection, silence analysis)

---

### Dimension 5 — Visual Instructional Support (weight: 10%)

**Why 10%?**  
Slides and visual aids are valuable but not universally required.  Some excellent lessons rely entirely on verbal explanation.

**What is assessed:**
- Whether displayed material is relevant to the topic being explained
- Whether important on-screen text or diagrams are readable in the sampled frames
- Whether visual transitions cause confusion (e.g. next slide shown before topic is finished)

**Important distinctions:**
- An audio-only recording scores this dimension as UNAVAILABLE — it is not penalised as zero
- Absence of slides does not automatically lower this score if the lesson is effective without them
- This dimension is assessed on sampled frames only; full visual coverage is not claimed

**Primary evidence source:** video frames (described by GPT-4o vision)

---

## Scoring Formula

```
overall_score = Σ(dimension_score_i × weight_i) / Σ(weight_i for assessed dims)
```

Dimensions with status `unavailable` are excluded from both numerator and denominator.  Dimensions with status `partial` contribute at their full weight.

The Python scoring engine in `scoring.py` always performs this calculation.  The LLM never determines the final score or rating.

---

## Rating Thresholds

| Range | Rating | Interpretation |
|-------|--------|---------------|
| 0–39 | Poor | Significant teaching quality problems; not suitable for student use without revision |
| 40–59 | Average | Basic teaching present; notable gaps in clarity, engagement, or structure |
| 60–79 | Good | Competent teaching with some areas for improvement |
| 80–100 | Excellent | Strong teaching across most or all dimensions |

These thresholds are configurable and were not empirically calibrated.  They represent a reasonable first hypothesis for a 0–100 educational quality scale.

---

## Known Biases and Limitations

1. **Transcript-heavy**: Three of the five dimensions rely primarily on the transcript.  Poor audio quality lowers the transcript quality and therefore the scores for Clarity, Engagement, and Structure — even if the underlying teaching was good.

2. **No student signal**: Student learning is the ultimate measure of teaching quality.  Without student data, we use teacher behaviour as a proxy.  This is a necessary limitation of the available evidence.

3. **Frame sparsity**: A maximum of 12 frames are sampled.  A teacher who uses slides for the first half and a whiteboard for the second half may have the whiteboard portion unrepresented.

4. **LLM scoring subjectivity**: The AI evaluator interprets the evidence and assigns dimension scores.  Different temperature settings or model versions may produce slightly different scores for the same recording.

5. **Weight assumptions**: The 30/25/20/15/10 weight split is not validated against expert ratings.  A domain expert might weight Engagement higher or Audio lower for specific age groups or subjects.

6. **Language handling**: Whisper supports many languages, but the evaluation prompt is in English.  Non-English class recordings may produce lower-quality dimension explanations.

---

## What the System Will Not Do

- Invent transcript quotations, timestamps, or student responses
- Judge teachers based on appearance, accent, personality, or demographics
- Infer student learning outcomes without supporting evidence
- Claim to have analysed the full visual content when only frames were sampled
- Return a fabricated assessment when the evidence is insufficient
- Allow the AI model to change dimension weights or override the Python-calculated rating

# Day 29 – AI Conversation Flow Design

## Objective

Define how the AI interacts dynamically during automated screening calls, including decision-tree routing, edge-case handling (silence, confusion, repeated answers), fallback questions, follow-up triggers, and polite failure/retry logic.

---

## 1. AI Call Decision Tree

```
                          ┌─────────────────────┐
                          │  CALL INITIATED     │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  INTRO & CONSENT    │
                          │  "Is this a good    │
                          │   time to talk?"    │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
     │ YES / AVAILABLE │   │  SILENCE / NO   │   │  DECLINES /     │
     │                 │   │  RESPONSE       │   │  BUSY           │
     └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
              │                     │                     │
              ▼                     ▼                     ▼
     ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
     │ ASK CORE        │   │ REPROMPT ONCE   │   │ OFFER RESCHEDULE│
     │ SCREENING       │   │ (fallback Q)    │   │ → END CALL      │
     │ QUESTIONS       │   │                 │   │                 │
     └────────┬────────┘   └────────┬────────┘   └─────────────────┘
              │                     │
     ┌────────┴────────┐            │
     │                 │            │
     ▼                 ▼            ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│ ANSWER    │   │ STILL NO  │   │ REPROMPT  │
│ VALIDATED │   │ RESPONSE  │   │ 2ND TIME  │
└─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │               │
      ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│ FOLLOW-UP │   │ MARK      │   │ ESCALATE  │
│ TRIGGERED │   │ SKIPPED   │   │ TO HUMAN  │
│ BASED ON  │   │ / FAILED  │   │ / END     │
│ ANSWER    │   │           │   │           │
└─────┬─────┘   └─────┬─────┘   └─────┬─────┘
      │               │               │
      ▼               ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│ NEXT Q IN │   │ CONTINUE  │   │ LOG &     │
│ TREE      │   │ CALL      │   │ CLOSE     │
└───────────┘   └───────────┘   └───────────┘
```

### 1.1 Decision Tree Rules

| Condition | Action | Next State |
|---|---|---|
| Candidate answers clearly and correctly | Validate → score → route to next node | `awaiting_answer` → `answered` |
| Candidate answers but answer is ambiguous | Ask clarification fallback question | `needs_clarification` |
| Candidate says "I don't know" / "not sure" | Mark as skipped, continue | `skipped` |
| Silence for > 3 seconds (first occurrence) | Polite reprompt with shorter fallback question | `reprompting` |
| Silence again after reprompt | Mark question as skipped, move on | `skipped` |
| Repeated answer (same content as previous) | Acknowledge, ask follow-up probe | `needs_followup` |
| Hard filter fails (e.g., no required qualification) | Escalate to human review or end gracefully | `escalated` / `ended` |
| Candidate declines / asks to reschedule | Offer reschedule slot, end call politely | `rescheduling` → `ended` |
| Call drops mid-conversation | Resume from last answered question on retry | `interrupted` |

---

## 2. Edge-Case Handling

### 2.1 Silence Handling

**Detection:** No speech activity detected for 3 seconds after the AI finishes speaking.

| Attempt | AI Action | Example |
|---|---|---|
| 1st silence | Soft reprompt | "No worries — take your time. Could you tell me about your experience with React?" |
| 2nd silence | Shorter fallback question | "Have you worked with React before — yes or no?" |
| 3rd silence | Mark skipped, move on | "Okay, let's move to the next question." |

**Policy:**
- Max 2 reprompts per question
- Max 3 skipped questions per call before ending early
- Never repeat the same prompt verbatim more than twice

### 2.2 Confusion Handling

**Detection:** Candidate says "sorry?", "what do you mean?", "I don't understand", or asks for repetition.

| Attempt | AI Action | Example |
|---|---|---|
| 1st confusion | Rephrase the question more simply | "Let me put it another way: how many years have you worked as a software developer?" |
| 2nd confusion | Offer a yes/no fallback | "Have you worked as a developer for at least 2 years — yes or no?" |
| 3rd confusion | Skip and move on | "That's okay, let's continue with the next question." |

**Policy:**
- Max 2 rephrases per question
- Fallback to closed-ended (yes/no) questions when open-ended ones fail
- Log confusion events for call quality analytics

### 2.3 Repeated Answers Handling

**Detection:** Candidate repeats the same answer content (semantic similarity > 0.85) across 2+ turns.

**AI Action:**
1. Acknowledge the repetition politely
2. Ask a targeted follow-up probe to elicit new information
3. If repetition continues, mark as complete and move on

**Example:**
```
AI: "You mentioned you have 3 years of experience. Can you tell me about a specific project you worked on?"
Candidate: "I have 3 years of experience."
AI: "Thanks, I have that. To understand your experience better, what was the most challenging project you handled in those 3 years?"
```

**Policy:**
- Max 2 follow-up probes per repeated-answer cluster
- After 2 probes, accept the answer and continue

---

## 3. Fallback Questions

Fallback questions are simpler, closed-ended alternatives triggered when the primary question fails.

| Primary Question | Fallback Question | Trigger |
|---|---|---|
| "Tell me about yourself." | "Could you share your current role and years of experience?" | Silence / confusion |
| "Describe your experience with React." | "Have you worked with React — yes or no?" | Confusion / repeated answer |
| "What is your expected salary?" | "Is your expected salary within the range of ₹X–₹Y?" | Silence / confusion |
| "When can you join?" | "Are you available to join within 30 days?" | Confusion |
| "Why do you want this role?" | "Are you interested in this role — yes or no?" | Silence |

**Fallback design rules:**
- Always simpler and shorter than the primary question
- Prefer yes/no or single-value answers
- Preserve the same information goal as the primary question
- Never ask more than 2 fallback levels deep

---

## 4. Follow-Up Triggers

Follow-up questions are triggered dynamically based on the candidate's answer.

| Trigger Condition | Follow-Up Question | Purpose |
|---|---|---|
| Candidate mentions a specific skill | "Can you describe a project where you used that skill?" | Depth validation |
| Candidate claims N years of experience | "What was your most challenging project in that time?" | Experience verification |
| Candidate says "team lead" or "manager" | "How many people did you manage?" | Leadership validation |
| Candidate mentions a gap in employment | "Could you tell me about that gap?" | Gap clarification |
| Candidate's salary expectation exceeds budget | "Is there any flexibility in your expected salary?" | Negotiation signal |
| Candidate gives a short/vague answer | "Could you elaborate on that?" | Completeness check |
| Candidate asks about the role/company | Answer briefly, then return to screening | Engagement |

**Trigger priority order:**
1. Hard filter validation (eligibility)
2. Answer completeness
3. Depth probing (experience/skills)
4. Engagement (candidate questions)

---

## 5. Polite Failure & Retry Logic

### 5.1 Per-Question Failure Ladder

```
Primary Question
    │
    ├── Answer received → Validate → Route
    │
    ├── Silence → Reprompt 1 → Reprompt 2 → Skip → Continue
    │
    ├── Confusion → Rephrase 1 → Rephrase 2 → Skip → Continue
    │
    └── Invalid answer → Clarify → Fallback → Skip → Continue
```

### 5.2 Call-Level Failure Ladder

| Failure Event | Action |
|---|---|
| 3 consecutive skipped questions | End call early, mark as "incomplete" |
| Candidate hangs up | Log as "abandoned", offer retry in 1 hour |
| Call connection fails | Retry up to configured max attempts |
| AI service error | Escalate to human recruiter, log error |
| Candidate requests human | Warm transfer to recruiter, end AI flow |

### 5.3 Polite Exit Scripts

**Early termination (too many skipped):**
> "Thank you for your time today. I wasn't able to complete all the questions, but I've recorded your responses. A member of our team may follow up if needed. Have a great day."

**Reschedule request:**
> "No problem at all. I can help you reschedule. Would you prefer tomorrow morning or afternoon?"

**Decline to proceed:**
> "I understand, and that's completely fine. Thank you for your time, and we wish you all the best."

**Call drop:**
> "It looks like we may have been disconnected. I'll try reaching you again shortly, or you can call us back at your convenience."

---

## 6. State Machine Summary

The conversation state machine is implemented in:
`day29/conversation_state_machine.py`

Key states:
- `idle` — call not started
- `intro` — greeting and consent
- `awaiting_answer` — AI asked a question, waiting for response
- `answered` — answer received and validated
- `reprompting` — silence/confusion detected, re-asking
- `needs_clarification` — answer ambiguous
- `needs_followup` — follow-up triggered
- `skipped` — question skipped
- `escalated` — handed to human
- `rescheduling` — candidate wants a different time
- `ended` — call complete
- `failed` — call failed

See `day29/ai_call_flow_logic.md` for the full decision tree, edge-case handling, fallback questions, follow-up triggers, and retry logic.

---

## 7. Integration Points

| Component | Integration |
|---|---|
| `hr_screening_question_bank/3_ai_conversation_ready_objects.json` | Source of question nodes, validation rules, and routing |
| `day26_screening_scoring_engine/` | Receives validated answers for scoring |
| `day27_confidence_sentiment/` | Provides confidence/sentiment signals for follow-up triggers |
| `day28_screening_report_generator/` | Receives final call summary and skipped-question log |
| `eligibility_decision_engine/` | Receives hard-filter outcomes |

---

## 8. Configuration

Default thresholds (override per job role):

```yaml
silence_timeout_seconds: 3
max_reprompts_per_question: 2
max_confusion_reprompts: 2
max_skipped_questions_per_call: 3
retry_after_seconds: 3600   # 1 hour
max_call_retry_attempts: 3
semantic_similarity_repeat_threshold: 0.85
```

---

## 9. Open Questions / Future Work

- Support for multilingual fallback phrasing (currently English-only examples)
- Warm-transfer integration with live recruiters
- Real-time sentiment-based tone adjustment
- Candidate-initiated questions during the call
- Post-call SMS/WhatsApp follow-up (future scope per PRD Phase 8)

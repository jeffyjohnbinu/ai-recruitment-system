"""
Screening report builder.

Builds a structured recruiter-facing screening report from ATS results
and screening/interview answers.
"""

import re
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional


class _AnswerExtractor:
    """Extract structured information from candidate answers."""

    KEY_QUESTION_IDS: ClassVar[Dict[str, str]] = {
        "salary": "salary_expectation",
        "notice": "notice_period",
        "experience": "experience",
        "location": "location",
        "relocation": "relocation",
        "availability": "availability",
    }

    SKILL_KEYWORDS: ClassVar[Dict[str, List[str]]] = {
        "python": ["python"],
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "opencv": ["opencv"],
        "yolo": ["yolo", "yolov8", "yolov5"],
        "computer vision": ["computer vision"],
        "nlp": ["nlp", "natural language processing"],
        "generative ai": ["generative ai", "genai", "gen ai"],
        "llm": ["llm", "large language model"],
        "rag": ["rag", "retrieval augmented generation"],
        "langchain": ["langchain"],
        "sql": ["sql"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "docker": ["docker"],
        "aws": ["aws", "amazon web services"],
        "azure": ["azure"],
        "gcp": ["gcp", "google cloud"],
        "git": ["git", "github"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
    }

    @staticmethod
    def _normalise_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def extract_salary_expectation(text: str) -> Optional[float]:
        """Extract expected salary in LPA."""

        text = _AnswerExtractor._normalise_text(text)

        if not text:
            return None

        # Example: "20 to 25 LPA" -> 22.5
        range_match = re.search(
            r"(\d+(?:\.\d+)?)\s*" r"(?:-|to)\s*" r"(\d+(?:\.\d+)?)\s*" r"(lpa|lakhs?|lacs?|lac)?",
            text,
        )

        if range_match:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return round((low + high) / 2, 2)

        # Example: "15 lakhs per annum"
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*" r"(lpa|lakhs?|lacs?|lac)\b",
            text,
        )

        if match:
            return float(match.group(1))

        # Example: ₹15 LPA
        match = re.search(
            r"₹\s*(\d+(?:\.\d+)?)\s*" r"(lpa|lakhs?|lacs?|lac)\b",
            text,
        )

        if match:
            return float(match.group(1))

        # Example: 50k = 0.05 LPA
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*k\b",
            text,
        )

        if match:
            return float(match.group(1)) * 0.001

        return None

    @staticmethod
    def extract_notice_period(text: str) -> Optional[int]:
        """Extract notice period in days."""

        text = _AnswerExtractor._normalise_text(text)

        if not text:
            return None

        if any(
            phrase in text
            for phrase in [
                "immediate",
                "immediately",
                "can join now",
                "available now",
            ]
        ):
            return 0

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:month|months)",
            text,
        )

        if match:
            return int(float(match.group(1)) * 30)

        match = re.search(
            r"(\d+)\s*(?:week|weeks)",
            text,
        )

        if match:
            return int(match.group(1)) * 7

        match = re.search(
            r"(\d+)\s*(?:day|days)",
            text,
        )

        if match:
            return int(match.group(1))

        return None

    @staticmethod
    def extract_experience(text: str) -> Optional[float]:
        """Extract years of professional experience."""

        text = _AnswerExtractor._normalise_text(text)

        if not text:
            return None

        if any(
            phrase in text
            for phrase in [
                "i am a fresher",
                "i'm a fresher",
                "i am fresher",
                "i'm fresher",
                "fresh graduate",
                "recent graduate",
            ]
        ):
            return 0.0

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
            text,
        )

        if match:
            return float(match.group(1))

        return None

    @staticmethod
    def extract_location(text: str) -> Optional[str]:
        """Extract a commonly mentioned location."""

        text = _AnswerExtractor._normalise_text(text)

        if not text:
            return None

        locations = [
            "kochi",
            "cochin",
            "thiruvananthapuram",
            "trivandrum",
            "thrissur",
            "kottayam",
            "kollam",
            "calicut",
            "kozhikode",
            "bangalore",
            "bengaluru",
            "chennai",
            "hyderabad",
            "pune",
            "mumbai",
            "delhi",
            "noida",
            "gurgaon",
            "gurugram",
            "ahmedabad",
            "kolkata",
            "jaipur",
            "kerala",
            "tamil nadu",
            "karnataka",
            "telangana",
            "maharashtra",
        ]

        for location in locations:
            if location in text:
                return location.title()

        return None

    @staticmethod
    def extract_relocation_willingness(
        text: str,
    ) -> Optional[bool]:
        """Extract relocation willingness."""

        text = _AnswerExtractor._normalise_text(text)

        if not text:
            return None

        negative_phrases = [
            "not willing",
            "cannot relocate",
            "can't relocate",
            "do not want to relocate",
            "don't want to relocate",
            "not open to relocation",
            "not willing to relocate",
            "no, not",
        ]

        if any(phrase in text for phrase in negative_phrases):
            return False

        positive_phrases = [
            "willing to relocate",
            "open to relocation",
            "open to relocate",
            "can relocate",
            "ready to relocate",
            "yes, willing",
            "yes willing",
            "yes, relocate",
            "yes relocate",
        ]

        if any(phrase in text for phrase in positive_phrases):
            return True

        if text in {"yes", "y", "sure", "okay", "ok"}:
            return True

        if text in {"no", "n"}:
            return False

        return None

    @staticmethod
    def extract_relocation(text: str) -> Optional[bool]:
        """Backward-compatible relocation alias."""

        return _AnswerExtractor.extract_relocation_willingness(text)

    @staticmethod
    def extract_availability(text: str) -> Optional[str]:
        """Extract availability category."""

        text = _AnswerExtractor._normalise_text(text)

        if not text:
            return None

        if any(
            phrase in text
            for phrase in [
                "immediate",
                "immediately",
                "available now",
                "can join now",
            ]
        ):
            return "immediate"

        if any(
            phrase in text
            for phrase in [
                "within a month",
                "within one month",
                "in a month",
                "one month",
            ]
        ):
            return "within_month"

        if any(
            phrase in text
            for phrase in [
                "within two months",
                "within 2 months",
                "in two months",
                "in 2 months",
            ]
        ):
            return "within_2_months"

        if any(
            phrase in text
            for phrase in [
                "not sure",
                "unsure",
                "uncertain",
                "don't know",
                "do not know",
                "not certain",
            ]
        ):
            return "uncertain"

        match = re.search(
            r"(\d+)\s*(?:week|weeks)",
            text,
        )

        if match:
            return f"{match.group(1)}_weeks"

        match = re.search(
            r"(\d+)\s*(?:month|months)",
            text,
        )

        if match:
            return f"{match.group(1)}_months"

        match = re.search(
            r"(\d+)\s*(?:day|days)",
            text,
        )

        if match:
            return f"{match.group(1)}_days"

        return text

    @classmethod
    def extract_skills(cls, text: str) -> List[str]:
        """Extract known technical skills."""

        text = cls._normalise_text(text)

        if not text:
            return []

        skills = []

        for skill, keywords in cls.SKILL_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                skills.append(skill)

        return skills


class ScreeningReportBuilder:
    """Build a recruiter-facing screening report."""

    def __init__(
        self,
        ats_result: Any = None,
        screening_result: Any = None,
        job_rules: Optional[Dict[str, Any]] = None,
    ):
        self.ats_result = ats_result or {}
        self.screening_result = screening_result or {}
        self.job_rules = job_rules or {}

    def build_report(
        self,
        ats_result: Any = None,
        screening_result: Any = None,
        job_rules: Optional[Dict[str, Any]] = None,
        session_score: Optional[float] = None,
        **kwargs: Any,
    ):
        """Build the complete screening report."""

        ats = ats_result if ats_result is not None else self.ats_result

        screening = screening_result if screening_result is not None else self.screening_result

        rules = job_rules if job_rules is not None else self.job_rules

        if session_score is not None:
            screening = self._copy_with_session_score(
                screening,
                session_score,
            )

        return self._create_report(
            ats,
            screening,
            rules,
        )

    def _create_report(
        self,
        ats_result: Any,
        screening_result: Any,
        job_rules: Dict[str, Any],
    ):
        """Create a fully populated ScreeningReport."""

        from .report_format import (
            CompensationInsights,
            KeyAnswer,
            RiskProfile,
            ScreeningReport,
            SkillConfirmation,
            StrengthProfile,
        )

        ats = self._to_dict(ats_result)
        screening = self._to_dict(screening_result)

        answers = self._extract_key_answers(screening)

        strengths_raw = self._build_strength_profile(
            ats,
            screening,
            answers,
        )

        risks_raw = self._build_risk_profile(
            ats,
            screening,
            answers,
            job_rules,
        )

        compensation_raw = self._build_compensation_insights(
            answers,
            job_rules,
        )

        skills_raw = self._build_skill_confirmation(
            ats,
            screening,
            answers,
        )

        flags = self._build_flags(
            ats,
            screening,
            answers,
            job_rules,
        )

        confidence = self._compute_confidence(
            ats,
            screening,
            answers,
        )

        overall_score = self._extract_screening_score(screening) or self._extract_score(ats) or 0.0

        recommendation = self._recommend(
            overall_score,
            confidence,
            flags,
            risks_raw,
        )

        generated_at = datetime.now(timezone.utc).isoformat()

        candidate_id = self._first_value(
            screening,
            ats,
            [
                "candidate_id",
                "candidateId",
                "id",
            ],
            "unknown_candidate",
        )

        job_id = self._first_value(
            screening,
            ats,
            ["job_id", "jobId"],
            "unknown_job",
        )

        session_id = self._first_value(
            screening,
            ["session_id", "sessionId"],
            "unknown_session",
        )

        role_id = self._first_value(
            screening,
            ats,
            ["role_id", "roleId"],
            str(job_id),
        )

        request_id = self._first_value(
            screening,
            ats,
            ["request_id", "requestId"],
            f"day28-{candidate_id}",
        )

        session_score_data = (
            screening.get("session_score", {})
            if isinstance(screening.get("session_score"), dict)
            else {
                "overall_score": overall_score,
            }
        )

        behavioral_report = (
            screening.get("behavioral_report", {})
            if isinstance(screening.get("behavioral_report"), dict)
            else {}
        )

        key_answers = self._build_key_answer_objects(
            screening,
            answers,
        )

        strength_profile = StrengthProfile(
            communication_strength_score=0.0,
            clarity_score=0.0,
            confidence_score=confidence,
            conviction_score=0.0,
            engagement_score=0.0,
            professionalism_score=0.0,
            key_strengths=strengths_raw,
            development_areas=[],
        )

        risk_profile = RiskProfile(
            hesitation_level="minimal",
            uncertainty_level=(
                "moderate"
                if any("uncertain" in str(item).lower() for item in risks_raw)
                else "minimal"
            ),
            sentiment_risk="neutral",
            contradiction_count=0,
            red_flag_count=len(flags),
            critical_risks=risks_raw,
            warnings=[],
        )

        compensation_insights = CompensationInsights(
            salary_expectation=compensation_raw["expected_salary_lpa"],
            notice_period_days=answers["notice_period_days"],
            notice_period_qualifier=None,
            availability_status=(answers["availability"] or "not_specified"),
            salary_confidence=(
                "high" if answers["salary_expectation_lpa"] is not None else "uncertain"
            ),
            location_preferences=([answers["location"]] if answers["location"] else []),
            relocation_willingness=answers["willing_to_relocate"],
        )

        skill_confirmation = SkillConfirmation(
            confirmed_skills=skills_raw["screening_confirmed_skills"],
            skill_gaps=skills_raw["unconfirmed_ats_skills"],
            skill_confidence_scores={
                skill: 1.0 for skill in skills_raw["screening_confirmed_skills"]
            },
            experience_level=(answers["experience_level"] or "entry"),
            years_experience=answers["years_experience"],
        )

        recruiter_narrative = self._build_recruiter_narrative(
            ats,
            screening,
            answers,
            strengths_raw,
            risks_raw,
        )

        executive_summary = self._build_executive_summary(
            ats,
            answers,
            strengths_raw,
            risks_raw,
            confidence,
        )

        return ScreeningReport(
            candidate_id=str(candidate_id),
            job_id=str(job_id),
            session_id=str(session_id),
            role_id=str(role_id),
            generated_at=generated_at,
            request_id=str(request_id),
            session_score=session_score_data,
            behavioral_report=behavioral_report,
            key_answers=key_answers,
            strength_profile=strength_profile,
            risk_profile=risk_profile,
            compensation_insights=compensation_insights,
            skill_confirmation=skill_confirmation,
            overall_red_flags=flags,
            overall_warnings=[],
            recommendation=recommendation,
            confidence_score=confidence,
            overall_score=overall_score,
            recruiter_narrative=recruiter_narrative,
            executive_summary=executive_summary,
        )

    def _copy_with_session_score(
        self,
        screening_result: Any,
        session_score: float,
    ) -> Dict[str, Any]:
        """Add session score without mutating input."""

        data = dict(self._to_dict(screening_result))
        data["session_score"] = {
            "overall_score": session_score,
            "score": session_score,
        }

        return data

    def _extract_key_answers(
        self,
        screening_result: Any,
    ) -> Dict[str, Any]:
        """Extract important candidate answers."""

        result = self._to_dict(screening_result)

        answers = result.get(
            "answers",
            result,
        )

        if not isinstance(answers, dict):
            answers = {}

        combined_text = self._flatten_text(answers)

        salary_text = self._find_answer(
            answers,
            [
                "salary",
                "salary_expectation",
                "expected_salary",
                "compensation",
            ],
        )

        notice_text = self._find_answer(
            answers,
            [
                "notice",
                "notice_period",
                "notice_period_days",
            ],
        )

        experience_text = self._find_answer(
            answers,
            [
                "experience",
                "years_experience",
                "work_experience",
            ],
        )

        location_text = self._find_answer(
            answers,
            [
                "location",
                "current_location",
                "city",
            ],
        )

        relocation_text = self._find_answer(
            answers,
            [
                "relocation",
                "willing_to_relocate",
                "relocation_willingness",
            ],
        )

        availability_text = self._find_answer(
            answers,
            [
                "availability",
                "joining",
                "joining_date",
            ],
        )

        salary = (
            _AnswerExtractor.extract_salary_expectation(str(salary_text))
            if salary_text is not None
            else None
        )

        if salary is None:
            salary = _AnswerExtractor.extract_salary_expectation(combined_text)

        notice = (
            _AnswerExtractor.extract_notice_period(str(notice_text))
            if notice_text is not None
            else None
        )

        if notice is None:
            notice = _AnswerExtractor.extract_notice_period(combined_text)

        experience = (
            _AnswerExtractor.extract_experience(str(experience_text))
            if experience_text is not None
            else None
        )

        if experience is None:
            experience = _AnswerExtractor.extract_experience(combined_text)

        location = (
            _AnswerExtractor.extract_location(str(location_text))
            if location_text is not None
            else None
        )

        if location is None:
            location = _AnswerExtractor.extract_location(combined_text)

        relocation = (
            _AnswerExtractor.extract_relocation_willingness(str(relocation_text))
            if relocation_text is not None
            else None
        )

        if relocation is None:
            relocation = _AnswerExtractor.extract_relocation_willingness(combined_text)

        availability = (
            _AnswerExtractor.extract_availability(str(availability_text))
            if availability_text is not None
            else None
        )

        if availability is None:
            availability = _AnswerExtractor.extract_availability(combined_text)

        skills = _AnswerExtractor.extract_skills(combined_text)

        experience_level = None

        if experience is not None:
            if experience >= 8:
                experience_level = "lead"
            elif experience >= 5:
                experience_level = "senior"
            elif experience >= 3:
                experience_level = "mid"
            elif experience > 0:
                experience_level = "junior"
            else:
                experience_level = "entry"

        return {
            "salary_expectation_lpa": salary,
            "notice_period_days": notice,
            "years_experience": experience,
            "experience_level": experience_level,
            "location": location,
            "willing_to_relocate": relocation,
            "availability": availability,
            "confirmed_skills": skills,
        }

    def _build_strength_profile(
        self,
        ats_result: Any,
        screening_result: Any,
        answers: Dict[str, Any],
    ) -> List[str]:
        """Build candidate strengths."""

        strengths = []

        ats_score = self._extract_score(self._to_dict(ats_result))

        if ats_score is not None:
            if ats_score >= 0.80:
                strengths.append("Strong ATS alignment with the target role.")
            elif ats_score >= 0.65:
                strengths.append("Good ATS alignment with the target role.")

        skills = answers.get(
            "confirmed_skills",
            [],
        )

        if skills:
            strengths.append("Demonstrates familiarity with " + ", ".join(skills[:6]) + ".")

        experience = answers.get("years_experience")

        if experience is not None:
            if experience >= 3:
                strengths.append(
                    f"Has approximately {experience:g} years " "of relevant experience."
                )
            elif experience > 0:
                strengths.append("Has prior practical experience relevant " "to the role.")
            else:
                strengths.append(
                    "Recent graduate/fresher profile with " "opportunity for structured onboarding."
                )

        screening_score = self._extract_screening_score(self._to_dict(screening_result))

        if screening_score is not None:
            if screening_score >= 0.80:
                strengths.append("Strong screening performance.")
            elif screening_score >= 0.65:
                strengths.append("Positive screening performance.")

        if not strengths:
            strengths.append("Candidate information is available " "for recruiter review.")

        return strengths

    def _build_risk_profile(
        self,
        ats_result: Any,
        screening_result: Any,
        answers: Dict[str, Any],
        job_rules: Dict[str, Any],
    ) -> List[str]:
        """Build candidate risk observations."""

        risks = []

        ats = self._to_dict(ats_result)

        ats_score = self._extract_score(ats)

        minimum_ats = self._safe_float(job_rules.get("minimum_ats_score"))

        if ats_score is not None and minimum_ats is not None and ats_score < minimum_ats:
            risks.append("ATS score is below the configured " "minimum threshold.")

        required_experience = self._safe_float(job_rules.get("minimum_experience"))

        experience = self._safe_float(answers.get("years_experience"))

        if (
            required_experience is not None
            and experience is not None
            and experience < required_experience
        ):
            risks.append("Experience is below the configured " "job requirement.")

        mandatory_skills = job_rules.get(
            "mandatory_skills",
            [],
        )

        if isinstance(mandatory_skills, str):
            mandatory_skills = [mandatory_skills]

        confirmed_skills = {
            skill.lower()
            for skill in answers.get(
                "confirmed_skills",
                [],
            )
        }

        missing_skills = [
            skill for skill in mandatory_skills if str(skill).lower() not in confirmed_skills
        ]

        if missing_skills:
            risks.append(
                "Mandatory skills requiring verification: "
                + ", ".join(map(str, missing_skills))
                + "."
            )

        if (
            job_rules.get("relocation_required") is True
            and answers.get("willing_to_relocate") is False
        ):
            risks.append("Candidate is not willing to relocate " "while relocation is required.")

        screening_score = self._extract_screening_score(self._to_dict(screening_result))

        if screening_score is not None and screening_score < 0.50:
            risks.append(
                "Screening performance indicates areas " "requiring additional evaluation."
            )

        if not risks:
            risks.append(
                "No major risk indicators were identified " "from the available screening data."
            )

        return risks

    def _build_compensation_insights(
        self,
        answers: Dict[str, Any],
        job_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build compensation analysis."""

        expected = self._safe_float(answers.get("salary_expectation_lpa"))

        minimum = self._safe_float(job_rules.get("minimum_salary_lpa"))

        maximum = self._safe_float(job_rules.get("maximum_salary_lpa"))

        status = "not_provided"

        if expected is not None:
            if maximum is not None and expected > maximum:
                status = "above_range"
            elif minimum is not None and expected < minimum:
                status = "below_range"
            else:
                status = "within_range"

        return {
            "expected_salary_lpa": expected,
            "minimum_salary_lpa": minimum,
            "maximum_salary_lpa": maximum,
            "status": status,
        }

    def _build_skill_confirmation(
        self,
        ats_result: Any,
        screening_result: Any,
        answers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compare ATS skills with screening-confirmed skills."""

        ats = self._to_dict(ats_result)
        screening = self._to_dict(screening_result)

        ats_skills = self._extract_ats_skills(ats)

        screening_text = self._flatten_text(screening)

        confirmed = _AnswerExtractor.extract_skills(screening_text)

        if answers:
            confirmed.extend(
                answers.get(
                    "confirmed_skills",
                    [],
                )
            )

        ats_normalised = {str(skill).strip().lower() for skill in ats_skills}

        confirmed_normalised = {str(skill).strip().lower() for skill in confirmed}

        matched = sorted(ats_normalised.intersection(confirmed_normalised))

        unconfirmed = sorted(ats_normalised.difference(confirmed_normalised))

        newly_confirmed = sorted(confirmed_normalised.difference(ats_normalised))

        return {
            "ats_skills": sorted(ats_normalised),
            "screening_confirmed_skills": sorted(confirmed_normalised),
            "matched_skills": matched,
            "unconfirmed_ats_skills": unconfirmed,
            "newly_confirmed_skills": newly_confirmed,
        }

    def _build_flags(
        self,
        ats_result: Any,
        screening_result: Any,
        answers: Dict[str, Any],
        job_rules: Dict[str, Any],
    ) -> List[str]:
        """Build recruiter review flags."""

        flags = []

        ats = self._to_dict(ats_result)

        ats_score = self._extract_score(ats)

        if ats_score is not None and ats_score < 0.50:
            flags.append("LOW_ATS_SCORE")

        compensation = self._build_compensation_insights(
            answers,
            job_rules,
        )

        if compensation["status"] == "above_range":
            flags.append("SALARY_ABOVE_RANGE")

        if compensation["status"] == "below_range":
            flags.append("SALARY_BELOW_RANGE")

        if answers.get("willing_to_relocate") is False:
            flags.append("RELOCATION_RESTRICTION")

        if answers.get("salary_expectation_lpa") is None:
            flags.append("SALARY_NOT_PROVIDED")

        if answers.get("years_experience") is None:
            flags.append("EXPERIENCE_NOT_PROVIDED")

        return list(dict.fromkeys(flags))

    def _compute_confidence(
        self,
        ats_result: Any,
        screening_result: Any,
        answers: Dict[str, Any],
    ) -> float:
        """Compute confidence in generated report."""

        ats = self._to_dict(ats_result)
        screening = self._to_dict(screening_result)

        available = 0

        if self._extract_score(ats) is not None:
            available += 1

        if self._extract_screening_score(screening) is not None:
            available += 1

        fields = [
            "salary_expectation_lpa",
            "notice_period_days",
            "years_experience",
            "location",
            "willing_to_relocate",
            "availability",
        ]

        populated = sum(answers.get(field) is not None for field in fields)

        answer_confidence = populated / len(fields)

        score_confidence = available / 2

        confidence = score_confidence * 0.40 + answer_confidence * 0.60

        return round(
            max(0.0, min(1.0, confidence)),
            3,
        )

    def _build_recruiter_narrative(
        self,
        ats_result: Any,
        screening_result: Any,
        answers: Dict[str, Any],
        strengths: List[str],
        risks: List[str],
    ) -> str:
        """Build recruiter narrative."""

        ats_score = self._extract_score(self._to_dict(ats_result))

        parts = []

        if ats_score is not None:
            parts.append(f"The candidate has an ATS alignment " f"score of {ats_score:.0%}.")

        experience = answers.get("years_experience")

        if experience is not None:
            parts.append(
                f"The candidate reports approximately " f"{experience:g} years of experience."
            )

        salary = answers.get("salary_expectation_lpa")

        if salary is not None:
            parts.append(f"Expected compensation is approximately " f"{salary:g} LPA.")

        if strengths:
            parts.append("Key strengths include " + " ".join(strengths[:3]))

        meaningful_risks = [risk for risk in risks if "No major risk" not in risk]

        if meaningful_risks:
            parts.append("Recruiter attention is recommended for " + " ".join(meaningful_risks[:2]))

        return (
            " ".join(parts)
            if parts
            else "Insufficient candidate data for a detailed " "recruiter narrative."
        )

    def _build_executive_summary(
        self,
        ats_result: Any,
        answers: Dict[str, Any],
        strengths: List[str],
        risks: List[str],
        confidence: float,
    ) -> str:
        """Build concise executive summary."""

        ats_score = self._extract_score(self._to_dict(ats_result))

        parts = []

        if ats_score is not None:
            parts.append(f"ATS score: {ats_score:.0%}.")

        experience = answers.get("years_experience")

        if experience is not None:
            parts.append(f"Experience: {experience:g} years.")

        salary = answers.get("salary_expectation_lpa")

        if salary is not None:
            parts.append(f"Expected salary: {salary:g} LPA.")

        parts.append(f"Report confidence: {confidence:.0%}.")

        return " ".join(parts)

    def _build_key_answer_objects(
        self,
        screening_result: Dict[str, Any],
        answers: Dict[str, Any],
    ) -> List[Any]:
        """Create report-format KeyAnswer objects."""

        from .report_format import KeyAnswer

        raw_answers = screening_result.get(
            "answers",
            {},
        )

        if not isinstance(raw_answers, dict):
            raw_answers = {}

        objects = []

        mapping = [
            (
                "salary_expectation",
                "salary",
                answers.get("salary_expectation_lpa"),
            ),
            (
                "notice_period",
                "availability",
                answers.get("notice_period_days"),
            ),
            (
                "experience",
                "experience",
                answers.get("years_experience"),
            ),
            (
                "location",
                "location",
                answers.get("location"),
            ),
            (
                "relocation",
                "relocation",
                answers.get("willing_to_relocate"),
            ),
            (
                "availability",
                "availability",
                answers.get("availability"),
            ),
        ]

        for question_id, category, extracted in mapping:
            raw = raw_answers.get(question_id, "")

            objects.append(
                KeyAnswer(
                    question_id=question_id,
                    category=category,
                    question_text=question_id.replace(
                        "_",
                        " ",
                    ).title(),
                    candidate_answer=str(raw),
                    score=1.0 if raw else 0.0,
                    is_mandatory=False,
                    was_answered=bool(raw),
                    extracted_data={
                        "value": extracted,
                    },
                    red_flags=[],
                    warnings=[],
                )
            )

        return objects

    @staticmethod
    def _recommend(
        overall_score: float,
        confidence: float,
        flags: List[str],
        risks: List[str],
    ) -> str:
        """Generate recruiter recommendation."""

        if confidence < 0.25:
            return "insufficient_data"

        if "LOW_ATS_SCORE" in flags:
            return "reject"

        if overall_score >= 0.75 and not flags:
            return "proceed"

        if overall_score >= 0.50:
            return "hold"

        return "reject"

    @staticmethod
    def _first_value(
        *sources: Any,
    ) -> Any:
        """Return the first matching value from dictionaries."""

        default = None

        for source in sources:
            if not isinstance(source, dict):
                continue

            keys = None

            if isinstance(source, dict):
                # Support calls where a key list and default are supplied.
                continue

        return default

    @staticmethod
    def _find_answer(
        answers: Dict[str, Any],
        keys: List[str],
    ) -> Any:
        """Find answer by possible field names."""

        for key in keys:
            if key in answers:
                return answers[key]

        return None

    @staticmethod
    def _extract_score(
        data: Dict[str, Any],
    ) -> Optional[float]:
        """Extract ATS score."""

        for key in [
            "ats_score",
            "match_score",
            "score",
            "final_score",
            "overall_score",
        ]:
            if key in data:
                value = ScreeningReportBuilder._safe_float(data[key])

                if value is None:
                    continue

                if value > 1:
                    value /= 100

                return max(
                    0.0,
                    min(1.0, value),
                )

        return None

    @staticmethod
    def _extract_screening_score(
        data: Dict[str, Any],
    ) -> Optional[float]:
        """Extract screening/session score."""

        session_score = data.get("session_score")

        if isinstance(session_score, dict):
            for key in [
                "overall_score",
                "score",
                "final_score",
            ]:
                if key in session_score:
                    value = ScreeningReportBuilder._safe_float(session_score[key])

                    if value is not None:
                        if value > 1:
                            value /= 100

                        return max(
                            0.0,
                            min(1.0, value),
                        )

        for key in [
            "session_score",
            "screening_score",
            "interview_score",
            "overall_score",
            "score",
            "final_score",
        ]:
            if key in data:
                value = ScreeningReportBuilder._safe_float(data[key])

                if value is None:
                    continue

                if value > 1:
                    value /= 100

                return max(
                    0.0,
                    min(1.0, value),
                )

        return None

    @staticmethod
    def _extract_ats_skills(
        data: Dict[str, Any],
    ) -> List[str]:
        """Extract skills from ATS result."""

        skills = []

        for key in [
            "skills",
            "matched_skills",
            "extracted_skills",
            "required_skills",
            "candidate_skills",
        ]:
            value = data.get(key)

            if isinstance(value, list):
                skills.extend(str(item) for item in value if item)

            elif isinstance(value, dict):
                skills.extend(str(item) for item in value.keys() if item)

            elif isinstance(value, str):
                skills.extend(item.strip() for item in value.split(",") if item.strip())

        return list(dict.fromkeys(skills))

    @staticmethod
    def _flatten_text(value: Any) -> str:
        """Flatten nested data to text."""

        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            return " ".join(ScreeningReportBuilder._flatten_text(item) for item in value.values())

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return " ".join(ScreeningReportBuilder._flatten_text(item) for item in value)

        return str(value)

    @staticmethod
    def _to_dict(value: Any) -> Dict[str, Any]:
        """Convert object to dictionary."""

        if value is None:
            return {}

        if isinstance(value, dict):
            return value

        to_dict = getattr(
            value,
            "to_dict",
            None,
        )

        if callable(to_dict):
            try:
                result = to_dict()

                if isinstance(result, dict):
                    return result
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return dict(value.__dict__)
            except Exception:
                pass

        return {}

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:
        """Safely convert to float."""

        if value is None or isinstance(value, bool):
            return None

        if isinstance(value, dict):
            for key in [
                "overall_score",
                "score",
                "value",
            ]:
                if key in value:
                    return ScreeningReportBuilder._safe_float(value[key])

            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

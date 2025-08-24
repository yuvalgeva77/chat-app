# app/api/services/response_templates.py
"""
Deterministic response templates for common recruiter queries.
These provide reliable, fact-based answers without model hallucination.
"""
from typing import List, Tuple
import re

from app.core.rag_index import get_facts


def _bullets(items):
    return "\n".join(f"- {it}" for it in items if it)


def _facts() -> dict:
    return get_facts() or {}


# ===================== Core Templates =====================

def build_contact_response() -> str:
    """Contact information from facts.json"""
    b = _facts().get("basics", {}) or {}
    c = (b.get("contact") or {})
    email = c.get("email")
    phone = c.get("phone")
    linkedin = c.get("linkedin")
    github = c.get("github")

    parts = []
    if email: parts.append(f"Email: {email}")
    if phone: parts.append(f"Phone: {phone}")
    if linkedin: parts.append(f"LinkedIn: {linkedin}")
    if github: parts.append(f"GitHub: {github}")

    return "\n".join(parts) if parts else "Contact details are not specified in my sources."


def build_summary_response() -> str:
    """Professional summary from facts.json"""
    f = _facts()
    b = f.get("basics", {}) or {}
    highlights = f.get("highlights", []) or []

    lines = [
        f"Name: {b.get('name', 'Youval')}",
        f"Title: {b.get('title')}" if b.get("title") else None,
        f"Location: {b.get('location')}" if b.get("location") else None,
        f"Work Authorization: {b.get('work_authorization')}" if b.get("work_authorization") else None,
        f"Availability: {b.get('availability')}" if b.get("availability") else None,
    ]

    # Add highlights
    if highlights:
        lines.append("\nKey Highlights:")
        lines.append(_bullets(highlights))

    return "\n".join([x for x in lines if x])


def build_experience_response() -> str:
    """Professional experience from facts.json"""
    f = _facts()
    exp = f.get("experience", []) or []
    if not exp:
        return "Experience details are not specified in my sources."

    lines = ["Professional Experience:\n"]
    for i, role in enumerate(exp):
        # Role header
        title = role.get('title', 'Role')
        company = role.get('company', 'Company')
        period = role.get('period', 'Period')
        lines.append(f"{title} - {company} ({period})")

        # Role type if specified
        if role.get('type'):
            lines.append(f"*{role['type']}*")

        # Key achievements
        achievements = role.get('key_achievements', [])
        if achievements:
            for achievement in achievements:
                lines.append(f"• {achievement}")

        # Add spacing between roles (except last one)
        if i < len(exp) - 1:
            lines.append("")

    return "\n".join(lines)


def build_skills_response() -> str:
    """Technical skills from facts.json"""
    f = _facts()
    skills = f.get("skills", {}) or {}
    if not skills:
        return "Skills are not specified in my sources."

    lines = ["Technical Skills:\n"]

    # Define order for better presentation
    skill_order = ["languages", "frameworks_tools", "cloud_data", "practices"]

    for category in skill_order:
        items = skills.get(category)
        if items:
            category_name = category.replace('_', ' & ').title()
            if isinstance(items, list):
                lines.append(f"{category_name}: {', '.join(items)}")
            else:
                lines.append(f"{category_name}: {items}")

    # Add any remaining categories not in the predefined order
    for category, items in skills.items():
        if category not in skill_order and items:
            category_name = category.replace('_', ' & ').title()
            if isinstance(items, list):
                lines.append(f"{category_name}: {', '.join(items)}")
            else:
                lines.append(f"{category_name}: {items}")

    return "\n".join(lines)


def build_projects_response() -> str:
    """Key projects and achievements from experience"""
    f = _facts()
    exp = f.get("experience", []) or []
    lines = ["Key Projects & Achievements:\n"]

    for role in exp:
        achievements = role.get('key_achievements', [])
        if achievements:
            company = role.get('company', 'Company')
            period = role.get('period', '')
            lines.append(f"At {company} {f'({period})' if period else ''}:")
            for achievement in achievements:
                lines.append(f"• {achievement}")
            lines.append("")  # Spacing between companies

    result = "\n".join(lines).strip()
    return result if len(lines) > 2 else "Project details are not specified in my sources."


def build_availability_response() -> str:
    """Availability information"""
    avail = (_facts().get("basics", {}) or {}).get("availability")
    return f"Availability: {avail or 'Not specified in my sources.'}"


def build_work_authorization_response() -> str:
    """Work authorization status"""
    auth = (_facts().get("basics", {}) or {}).get("work_authorization")
    return f"Work Authorization: {auth or 'Not specified in my sources.'}"


def build_location_response() -> str:
    """Location information"""
    loc = (_facts().get("basics", {}) or {}).get("location")
    return f"Location: {loc or 'Not specified in my sources.'}"


# ===================== Smart Hybrid Responses =====================

def build_smart_fallback(intent: str, user_message: str, retrieved_snippets: List[Tuple[str, str]]) -> str:
    """
    Intelligent fallback that combines templates with retrieved context for complex queries.
    This is used when the model generates poor responses.
    """

    # Get the base template response
    base_response = ""
    if intent == "experience":
        base_response = build_experience_response()
    elif intent == "projects":
        base_response = build_projects_response()
    elif intent == "skills":
        base_response = build_skills_response()
    elif intent == "contact":
        base_response = build_contact_response()
    else:
        base_response = build_summary_response()

    # If we have relevant retrieved snippets, add them as additional context
    if retrieved_snippets:
        relevant_context = []
        for title, text in retrieved_snippets[:2]:  # Top 2 most relevant
            # Only add substantial, non-duplicate content
            text_clean = text.strip()
            if len(text_clean) > 50 and not any(text_clean[:100] in base_response for _ in [1]):
                snippet = text_clean[:300] + "..." if len(text_clean) > 300 else text_clean
                relevant_context.append(f"Additional from {title}:\n{snippet}")

        if relevant_context:
            base_response += "\n\n" + "\n\n".join(relevant_context)

    return base_response

# ===================== Template Routing =====================

def get_template_response(intent: str) -> str:
    """Route intent to appropriate template response"""
    template_map = {
        "contact": build_contact_response,
        "summary": build_summary_response,
        "experience": build_experience_response,
        "skills": build_skills_response,
        "projects": build_projects_response,
        "availability": build_availability_response,
        "authorization": build_work_authorization_response,
        "location": build_location_response,
    }

    builder = template_map.get(intent)
    if builder:
        return builder()
    else:
        return build_summary_response()  # Default fallback


# ===================== Intent Detection (moved from chat_engine) =====================

CONTACT_RE = re.compile(
    r"\b(contact|contact info|contact information|reach (?:you|him|her)|email|e-?mail|mail|phone|phone number|telephone|how (?:can|do) i (?:reach|contact))\b",
    re.IGNORECASE)
SUMMARY_RE = re.compile(r"\b(summary|intro|about)\b", re.IGNORECASE)
AVAIL_RE = re.compile(r"\b(availability|available|start date|start immediately|when.*start)\b", re.IGNORECASE)
AUTH_RE = re.compile(r"\b(work (?:authorization|permit)|visa|work-authori[sz]ation)\b", re.IGNORECASE)
LOC_RE = re.compile(r"\b(location|based|where.*located|remote|hybrid|on[- ]?site)\b", re.IGNORECASE)
EXPERIENCE_RE = re.compile(r"\b(experience|background|roles?)\b", re.IGNORECASE)
PROJECTS_RE = re.compile(r"\b(projects?|portfolio|case stud(y|ies))\b", re.IGNORECASE)
SKILLS_RE = re.compile(r"\b(skills?|technologies|tech stack|programming languages?|tools?|frameworks?)\b",
                       re.IGNORECASE)
EDUCATION_RE = re.compile(r"\b(education|degree|university|college|studied?)\b", re.IGNORECASE)


def detect_intent(text: str) -> str:
    """Enhanced intent detection for recruiter queries"""
    s = text or ""
    if CONTACT_RE.search(s): return "contact"
    if SUMMARY_RE.search(s): return "summary"
    if AVAIL_RE.search(s): return "availability"
    if AUTH_RE.search(s): return "authorization"
    if LOC_RE.search(s): return "location"
    if SKILLS_RE.search(s): return "skills"
    if PROJECTS_RE.search(s): return "projects"
    if EXPERIENCE_RE.search(s): return "experience"
    if EDUCATION_RE.search(s): return "education"
    return "other"
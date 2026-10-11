
"""Evidence-based preliminary screening for hyperspectral Raman research."""

import re


def clean_text(value):
    """Normalize HTML and whitespace for keyword screening."""
    text = re.sub(r"<[^>]+>", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def has_any(text, patterns):
    """Match any regular-expression pattern."""
    return any(re.search(pattern, text) for pattern in patterns)


def matched_terms(text, terms):
    """Return terms whose patterns match the text."""
    return [term for term, pattern in terms if re.search(pattern, text)]


def screen_paper(paper):
    """Assign preliminary evidence labels; human verification is required."""
    title = str(paper.get("title", "") or "")
    abstract = str(paper.get("abstract", "") or "")
    title_text = clean_text(title)
    text = clean_text(f"{title} {abstract}")

    # Explicit method wording; SRS is recorded separately because it is
    # not necessarily equivalent to spontaneous Raman acquisition.
    hyper_raman_patterns = [
        r"\bhyperspectral\s+raman\b",
        r"\braman\s+hyperspectral\b",
        r"\bhyperspectral\s+stimulated\s+raman\b",
    ]
    explicit_hyper_raman = has_any(text, hyper_raman_patterns)
    title_explicit_hyper_raman = has_any(title_text, hyper_raman_patterns)

    biomedical_terms = [
        ("patient", r"\bpatients?\b"),
        ("clinical", r"\bclinical\b"),
        ("cancer", r"\bcancers?\b"),
        ("tumor", r"\btumou?rs?\b"),
        ("biopsy", r"\bbiops(?:y|ies)\b"),
        ("pathology", r"\bpatholog(?:y|ical|ist)\b"),
        ("renal", r"\brenal\b"),
        ("amyloid", r"\bamyloid(?:osis)?\b"),
        ("thyroid", r"\bthyroid\b"),
        ("breast", r"\bbreast\b"),
        ("brain", r"\bbrain\b"),
        ("glioblastoma", r"\bglioblastoma\b"),
        ("leukemia", r"\bleuk[ae]mia\b"),
        ("alzheimer", r"\balzheimer'?s?\b"),
        ("bladder", r"\bbladder\b"),
        ("carcinoma", r"\bcarcinoma\b"),
        ("follicular thyroid", r"\bfollicular thyroid\b"),
        ("human tissue", r"\bhuman tissue\b"),
        ("clinical specimen", r"\bclinical specimens?\b"),
        ("cell line", r"\bcell lines?\b"),
    ]
    diagnostic_phrases = [
        ("diagnosis", r"\bdiagnos(?:is|tic|tics|ed)\b"),
        ("disease classification", r"\bdisease classification\b"),
        ("cancer detection", r"\bcancer detection\b"),
        ("tumor classification", r"\btumou?r classification\b"),
        ("tumor margin assessment", r"\b(?:tumou?r|tissue) margins? assessment\b"),
        ("pathology", r"\bpatholog(?:y|ical)\b"),
        ("differentiation", r"\bdifferentiation\b"),
        ("automated recognition", r"\bautomated recognition\b"),
    ]
    ai_terms = [
        ("machine learning", r"\bmachine learning\b"),
        ("deep learning", r"\bdeep learning\b"),
        ("neural network", r"\bneural networks?\b"),
        ("artificial intelligence", r"\bartificial intelligence\b"),
        ("random forest", r"\brandom forests?\b"),
        ("support vector machine", r"\bsupport vector machines?\b"),
        ("transformer", r"\btransformers?\b"),
        ("convolutional neural network", r"\bconvolutional neural networks?\b"),
    ]
    analysis_terms = [
        ("chemometrics", r"\bchemometric\w*\b"),
        ("partial least squares", r"\bpartial least squares\b"),
        ("PCA", r"\bpca\b"),
    ]
    adjacent_terms = [
        ("stimulated Raman scattering", r"\bstimulated raman scattering\b"),
        ("coherent anti-Stokes Raman", r"\bcoherent anti[- ]stokes raman\b"),
        ("tip-enhanced Raman", r"\btip[- ]enhanced raman\b"),
        ("surface-enhanced Raman", r"\bsurface[- ]enhanced raman\b"),
    ]

    non_biomedical_application_terms = [
        r"\bmilk adulteration\b",
        r"\bmilk powder\b",
        r"\bdairy authentication\b",
        r"\bfood safety\b",
        r"\bfood adulteration\b",
        r"\bfruit and vegetables\b",
        r"\bplant cells?\b",
        r"\bplant tissues?\b",
        r"\bmicroplastics?\b",
        r"\bnanoplastics?\b",
        r"\bfood quality\b",
        r"\badulterants?\b",
    ]
    non_biomedical_application = has_any(
        text, non_biomedical_application_terms
    )

    bio_hits = matched_terms(text, biomedical_terms)
    diagnostic_hits = matched_terms(text, diagnostic_phrases)
    ai_hits = matched_terms(text, ai_terms)
    analysis_hits = matched_terms(text, analysis_terms)
    adjacent_hits = matched_terms(text, adjacent_terms)

    has_raman = bool(re.search(r"\braman\b", text))
    has_hyperspectral = bool(re.search(r"\bhyperspectral\b", text))
    has_biomedical = bool(bio_hits)
    has_diagnostic = bool(diagnostic_hits)
    is_srs = bool(re.search(r"\bstimulated raman scattering\b|\bsrs microscopy\b", text))

    # These are screening priorities, not verified scientific conclusions.

    if non_biomedical_application:
        label = "adjacent_method"
    elif title_explicit_hyper_raman and has_biomedical:
        label = "strong_candidate"
    elif has_raman and has_biomedical and (
        has_diagnostic or has_hyperspectral
    ):
        label = "potential_candidate"
    elif has_raman or has_hyperspectral or adjacent_hits:
        label = "adjacent_method"
    else:
        label = "out_of_scope"


    method_status = (
        "yes" if explicit_hyper_raman
        else "unclear" if has_raman and has_hyperspectral
        else "no"
    )
    biomedical_status = "yes" if has_biomedical else "no"
    ai_status = "yes" if ai_hits else "no"
    srs_status = "yes" if is_srs else "no"

    evidence = []
    if explicit_hyper_raman:
        evidence.append("explicit hyperspectral Raman wording")
    if bio_hits:
        evidence.append("biomedical terms: " + ", ".join(bio_hits))
    if diagnostic_hits:
        evidence.append("diagnostic phrases: " + ", ".join(diagnostic_hits))
    if ai_hits:
        evidence.append("AI/ML terms: " + ", ".join(ai_hits))
    if analysis_hits:
        evidence.append("analysis terms: " + ", ".join(analysis_hits))
    if adjacent_hits:
        evidence.append("related modalities: " + ", ".join(adjacent_hits))

    return {
        "pmid": str(paper.get("pmid", "")),
        "title": title,
        "year": paper.get("year", ""),
        "relevance_label": label,
        "uses_hyperspectral_raman": method_status,
        "biomedical": biomedical_status,
        "ai_ml": ai_status,
        "srs_mentioned": srs_status,
        "explicit_hyperspectral_raman_evidence": explicit_hyper_raman,
        "hyperspectral_raman_phrases": (
            "explicit wording found" if explicit_hyper_raman else ""
        ),
        "biomedical_evidence": "; ".join(bio_hits),
        "diagnostic_evidence": "; ".join(diagnostic_hits),
        "ai_or_analysis_evidence": "; ".join(ai_hits + analysis_hits),
        "related_modalities": "; ".join(adjacent_hits),
        "evidence_summary": " | ".join(evidence),
        "data_availability": "unknown",
        "verification_status": "requires_full_text_methods_review",
    }


def screen_papers(papers):
    """Screen a list of paper dictionaries."""
    return [screen_paper(paper) for paper in papers]

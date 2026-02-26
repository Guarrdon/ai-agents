"""
End-to-end tests for the Advanced AI Section documentation.

Covers:
- File existence (all expected docs are present)
- Document structure (required sections, heading hierarchy)
- Navigation link validity (internal markdown links resolve to real files)
- Breadcrumb navigation accuracy
- Content completeness (non-empty sections, minimum length)
- Sequential navigation (prev/next chains are consistent)
- README consistency (all docs listed in README exist on disk)
- Code block formatting
- Learning path ordering and cross-references
- Accessibility via heading hierarchy
- Last-updated dates present
"""

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repository layout constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
ADVANCED_DIR = REPO_ROOT / "advanced"
ADVANCED_DOCS_DIR = ADVANCED_DIR / "docs"
ADVANCED_README = ADVANCED_DIR / "README.md"
REPO_README = REPO_ROOT / "README.md"
BEGINNER_DIR = REPO_ROOT / "beginner"
BEGINNER_README = BEGINNER_DIR / "README.md"

# Main 8-topic learning path docs (ordered as declared in advanced/README.md)
MAIN_LEARNING_PATH = [
    "neural-networks.md",
    "deep-learning-architectures.md",
    "model-architectures.md",
    "large-language-models.md",
    "generative-ai.md",
    "prompt-engineering.md",
    "training-techniques.md",
    "fine-tuning.md",
    "evaluation.md",
    "ethics.md",
    "rag.md",
    "agents.md",
    "safety-alignment.md",
]

# Supplementary / numbered reference docs
SUPPLEMENTARY_DOCS = [
    "01-neural-networks.md",
    "02-deep-learning-architectures.md",
    "06-ai-ethics.md",
    "07-training-techniques.md",
    "08-model-evaluation.md",
]

# Outline / planning reference
OTHER_DOCS = ["outline.md"]

ALL_DOCS = MAIN_LEARNING_PATH + SUPPLEMENTARY_DOCS + OTHER_DOCS

# Core structured docs that follow strict learning-path format with all
# required sections. Excludes ethics.md and deep-learning-architectures.md
# which use an alternative blockquote-style format for some sections.
STRICTLY_STRUCTURED_DOCS = [
    "neural-networks.md",
    "model-architectures.md",
    "large-language-models.md",
    "generative-ai.md",
    "prompt-engineering.md",
    "training-techniques.md",
    "fine-tuning.md",
    "evaluation.md",
    "rag.md",
    "agents.md",
    "safety-alignment.md",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_code_blocks(text: str) -> str:
    """Remove all fenced code block content (```...```) from text.

    This prevents Python comments (# ...) and inline-code link patterns from
    being mistaken for Markdown headings or links during analysis.
    """
    return re.sub(r"```[^\n]*\n.*?```", "```\n```", text, flags=re.DOTALL)


def extract_markdown_links(text: str) -> list:
    """Return list of (link_text, href) for all markdown links.

    Only analyses text outside fenced code blocks to avoid picking up
    Python dictionary subscripts, NumPy/LaTeX notation, etc.
    """
    clean = strip_code_blocks(text)
    return re.findall(r"\[([^\]]+)\]\(([^)]+)\)", clean)


def is_local_file_link(href: str) -> bool:
    """Return True if the href is a relative path to a local file.

    Filters out:
    - HTTP(S) / mailto / anchor-only links
    - Hrefs containing characters that are not valid in file paths
      (e.g. math/LaTeX expressions like Z^[l])
    """
    if href.startswith(("http://", "https://", "mailto:", "#")):
        return False
    # Valid local file paths contain only: word chars, hyphens, dots, slashes
    if not re.match(r"^[\w\-./]+$", href):
        return False
    return True


def resolve_link(source_file: Path, href: str) -> Path:
    """Resolve a relative href from a source_file's directory."""
    href = href.split("#")[0]
    if not href:
        return source_file
    return (source_file.parent / href).resolve()


def get_headings(text: str) -> list:
    """Return list of (level, title) for all ATX headings outside code blocks.

    Excludes content inside fenced code blocks so that Python comments
    (# comment) are not misidentified as Markdown H1 headings.
    """
    headings = []
    in_code_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append((len(m.group(1)), m.group(2).strip()))
    return headings


# ---------------------------------------------------------------------------
# 1. File Existence Tests
# ---------------------------------------------------------------------------


class TestFileExistence:
    """All expected documentation files are present on disk."""

    def test_advanced_readme_exists(self):
        assert ADVANCED_README.exists(), "advanced/README.md is missing"

    def test_repo_readme_exists(self):
        assert REPO_README.exists(), "Root README.md is missing"

    def test_beginner_readme_exists(self):
        assert BEGINNER_README.exists(), "beginner/README.md is missing"

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_main_learning_path_doc_exists(self, doc):
        path = ADVANCED_DOCS_DIR / doc
        assert path.exists(), "Main learning path doc missing: docs/{}".format(doc)

    @pytest.mark.parametrize("doc", SUPPLEMENTARY_DOCS)
    def test_supplementary_doc_exists(self, doc):
        path = ADVANCED_DOCS_DIR / doc
        assert path.exists(), "Supplementary doc missing: docs/{}".format(doc)

    @pytest.mark.parametrize("doc", OTHER_DOCS)
    def test_other_doc_exists(self, doc):
        path = ADVANCED_DOCS_DIR / doc
        assert path.exists(), "Other doc missing: docs/{}".format(doc)

    def test_no_unexpected_docs(self):
        """Every .md file in advanced/docs/ is accounted for in our manifest."""
        on_disk = {f.name for f in ADVANCED_DOCS_DIR.glob("*.md")}
        expected = set(ALL_DOCS)
        unexpected = on_disk - expected
        assert not unexpected, (
            "Unexpected .md files found in advanced/docs/: {}\n"
            "Add them to the appropriate list in the test manifest.".format(unexpected)
        )


# ---------------------------------------------------------------------------
# 2. Document Structure Tests
# ---------------------------------------------------------------------------


class TestDocumentStructure:
    """Each document follows the required structural conventions."""

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_has_h1_title(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        h1s = [title for level, title in get_headings(text) if level == 1]
        assert h1s, "{}: No H1 title found".format(doc)

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_single_h1_title(self, doc):
        """Each document should have exactly one H1 heading (document title).

        Uses code-block-aware heading parser to avoid false positives from
        Python comments (# comment) inside fenced code blocks.
        """
        text = read(ADVANCED_DOCS_DIR / doc)
        h1s = [title for level, title in get_headings(text) if level == 1]
        assert len(h1s) == 1, (
            "{}: Expected exactly one H1 heading, found {}: {}".format(
                doc, len(h1s), h1s
            )
        )

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_has_learning_objectives_section(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "## Learning Objectives" in text or "Learning Objectives" in text, (
            "{}: Missing 'Learning Objectives' section".format(doc)
        )

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_has_prerequisites_section(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "## Prerequisites" in text or "Prerequisites" in text, (
            "{}: Missing 'Prerequisites' section".format(doc)
        )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_has_further_reading_section(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "further reading" in text.lower(), (
            "{}: Missing 'Further Reading' section".format(doc)
        )

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_has_key_concepts_summary(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "key concepts" in text.lower(), (
            "{}: Missing 'Key Concepts Summary' section".format(doc)
        )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_has_navigation_footer(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "Navigation:" in text or "Advanced Home" in text, (
            "{}: Missing navigation footer".format(doc)
        )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_has_breadcrumb(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        has_breadcrumb = "\U0001f4cd" in text or ("Home" in text and "Advanced" in text)
        assert has_breadcrumb, "{}: Missing breadcrumb navigation".format(doc)

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH + SUPPLEMENTARY_DOCS)
    def test_heading_hierarchy_no_skip(self, doc):
        """Headings must not skip levels (H1 to H3 without H2 is invalid).

        Uses code-block-aware heading parser to avoid false positives.
        """
        text = read(ADVANCED_DOCS_DIR / doc)
        headings = get_headings(text)
        prev_level = 0
        for level, title in headings:
            if prev_level > 0:
                assert level <= prev_level + 1, (
                    "{}: Heading level skips from H{} to H{} at '{}'".format(
                        doc, prev_level, level, title
                    )
                )
            prev_level = level

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_minimum_content_length(self, doc):
        """Docs should be substantive (> 1000 characters)."""
        text = read(ADVANCED_DOCS_DIR / doc)
        assert len(text) > 1000, (
            "{}: Document appears too short ({} chars).".format(doc, len(text))
        )

    def test_advanced_readme_has_learning_path_section(self):
        text = read(ADVANCED_README)
        assert "Learning Path" in text or "learning path" in text.lower(), (
            "advanced/README.md: Missing Learning Path section"
        )

    def test_advanced_readme_has_topics_section(self):
        text = read(ADVANCED_README)
        assert "## \U0001f4da Topics" in text or "## Topics" in text, (
            "advanced/README.md: Missing Topics section"
        )

    def test_advanced_readme_has_supplementary_section(self):
        text = read(ADVANCED_README)
        assert "Supplementary" in text, (
            "advanced/README.md: Missing Supplementary Content section"
        )

    def test_advanced_readme_has_home_link(self):
        text = read(ADVANCED_README)
        assert "[Home]" in text or "[← Back to Home]" in text, (
            "advanced/README.md: Missing link back to Home"
        )


# ---------------------------------------------------------------------------
# 3. Navigation Link Validity Tests
# ---------------------------------------------------------------------------


class TestNavigationLinks:
    """All local file links in every document resolve to real files.

    Uses a code-block-aware link extractor and a strict file-path filter to
    avoid false positives from LaTeX / math notation and Python subscript
    expressions inside code blocks.
    """

    @pytest.mark.parametrize("doc", ALL_DOCS)
    def test_all_local_links_resolve(self, doc):
        path = ADVANCED_DOCS_DIR / doc
        text = read(path)
        broken = []
        for link_text, href in extract_markdown_links(text):
            if not is_local_file_link(href):
                continue
            resolved = resolve_link(path, href)
            if not resolved.exists():
                broken.append("  [{}]({}) -> {}".format(link_text, href, resolved))
        assert not broken, (
            "{}: Broken local links found:\n".format(doc) + "\n".join(broken)
        )

    def test_advanced_readme_local_links(self):
        text = read(ADVANCED_README)
        broken = []
        for link_text, href in extract_markdown_links(text):
            if not is_local_file_link(href):
                continue
            resolved = resolve_link(ADVANCED_README, href)
            if not resolved.exists():
                broken.append("  [{}]({}) -> {}".format(link_text, href, resolved))
        assert not broken, (
            "advanced/README.md: Broken local links:\n" + "\n".join(broken)
        )

    def test_repo_readme_local_links(self):
        text = read(REPO_README)
        broken = []
        for link_text, href in extract_markdown_links(text):
            if not is_local_file_link(href):
                continue
            resolved = resolve_link(REPO_README, href)
            if not resolved.exists():
                broken.append("  [{}]({}) -> {}".format(link_text, href, resolved))
        assert not broken, (
            "README.md: Broken local links:\n" + "\n".join(broken)
        )


# ---------------------------------------------------------------------------
# 4. README Consistency Tests
# ---------------------------------------------------------------------------


class TestReadmeConsistency:
    """The advanced/README.md learning path references files that exist."""

    def test_all_main_path_docs_linked_in_advanced_readme(self):
        text = read(ADVANCED_README)
        missing = []
        for doc in MAIN_LEARNING_PATH:
            base = doc.replace(".md", "")
            if doc not in text and base not in text:
                missing.append(doc)
        assert not missing, (
            "advanced/README.md does not reference these main path docs: {}".format(
                missing
            )
        )

    def test_all_supplementary_docs_linked_in_advanced_readme(self):
        text = read(ADVANCED_README)
        missing = []
        for doc in SUPPLEMENTARY_DOCS:
            if doc not in text:
                missing.append(doc)
        assert not missing, (
            "advanced/README.md does not reference these supplementary docs: {}".format(
                missing
            )
        )

    def test_repo_readme_links_to_advanced_section(self):
        text = read(REPO_README)
        assert "./advanced/README.md" in text or "advanced/README.md" in text, (
            "Root README.md does not link to the Advanced section README"
        )

    def test_repo_readme_links_to_beginner_section(self):
        text = read(REPO_README)
        assert "./beginner/README.md" in text or "beginner/README.md" in text, (
            "Root README.md does not link to the Beginner section README"
        )

    def test_advanced_readme_links_to_beginner_section(self):
        text = read(ADVANCED_README)
        assert "beginner" in text.lower(), (
            "advanced/README.md does not reference the Beginner section"
        )

    def test_advanced_readme_learning_path_order_matches_main_path(self):
        """The learning path code block in the README lists topics in order."""
        text = read(ADVANCED_README)
        expected_topic_keywords = [
            "Neural Networks",
            "Deep Learning",
            "Model Architectures",
            "Large Language Models",
            "Generative AI",
            "Prompt Engineering",
            "Training Techniques",
            "Fine-Tuning",
            "Evaluation",
            "AI Ethics",
            "Retrieval-Augmented",
            "AI Agents",
            "Safety",
        ]
        positions = []
        for keyword in expected_topic_keywords:
            pos = text.find(keyword)
            assert pos >= 0, (
                "advanced/README.md: Expected topic '{}' not found in text".format(
                    keyword
                )
            )
            positions.append(pos)
        assert positions == sorted(positions), (
            "advanced/README.md: Topics are not listed in the expected sequential order"
        )


# ---------------------------------------------------------------------------
# 5. Sequential Navigation Consistency
# ---------------------------------------------------------------------------


class TestSequentialNavigation:
    """The prev/next navigation footer chain is internally consistent."""

    def _get_nav_footer(self, doc):
        """Extract the *Navigation: ...* footer line from a doc."""
        text = read(ADVANCED_DOCS_DIR / doc)
        for line in text.splitlines():
            if "Navigation:" in line:
                return line
        return None

    def test_neural_networks_next_is_deep_learning(self):
        footer = self._get_nav_footer("neural-networks.md")
        assert footer is not None, "neural-networks.md has no navigation footer"
        assert "deep-learning-architectures" in footer

    def test_deep_learning_prev_is_neural_networks(self):
        footer = self._get_nav_footer("deep-learning-architectures.md")
        assert footer is not None
        assert "neural-networks" in footer

    def test_deep_learning_next_is_model_architectures(self):
        footer = self._get_nav_footer("deep-learning-architectures.md")
        assert footer is not None
        assert "model-architectures" in footer

    def test_model_architectures_prev_is_deep_learning(self):
        footer = self._get_nav_footer("model-architectures.md")
        assert footer is not None
        assert "deep-learning-architectures" in footer

    def test_model_architectures_next_is_llm(self):
        footer = self._get_nav_footer("model-architectures.md")
        assert footer is not None
        assert "large-language-models" in footer

    def test_llm_prev_is_model_architectures(self):
        footer = self._get_nav_footer("large-language-models.md")
        assert footer is not None
        assert "model-architectures" in footer

    def test_llm_next_is_generative_ai(self):
        footer = self._get_nav_footer("large-language-models.md")
        assert footer is not None
        assert "generative-ai" in footer

    def test_generative_ai_prev_is_llm(self):
        footer = self._get_nav_footer("generative-ai.md")
        assert footer is not None
        assert "large-language-models" in footer

    def test_generative_ai_next_is_prompt_engineering(self):
        footer = self._get_nav_footer("generative-ai.md")
        assert footer is not None
        assert "prompt-engineering" in footer

    def test_prompt_engineering_prev_is_generative_ai(self):
        footer = self._get_nav_footer("prompt-engineering.md")
        assert footer is not None
        assert "generative-ai" in footer

    def test_prompt_engineering_next_is_training_techniques(self):
        footer = self._get_nav_footer("prompt-engineering.md")
        assert footer is not None
        assert "training-techniques" in footer

    def test_training_techniques_prev_is_prompt_engineering(self):
        footer = self._get_nav_footer("training-techniques.md")
        assert footer is not None
        assert "prompt-engineering" in footer

    def test_training_techniques_next_is_fine_tuning(self):
        footer = self._get_nav_footer("training-techniques.md")
        assert footer is not None
        assert "fine-tuning" in footer

    def test_fine_tuning_prev_is_training_techniques(self):
        footer = self._get_nav_footer("fine-tuning.md")
        assert footer is not None
        assert "training-techniques" in footer

    def test_fine_tuning_next_is_evaluation(self):
        footer = self._get_nav_footer("fine-tuning.md")
        assert footer is not None
        assert "evaluation" in footer

    def test_evaluation_prev_is_fine_tuning(self):
        footer = self._get_nav_footer("evaluation.md")
        assert footer is not None
        assert "fine-tuning" in footer

    def test_evaluation_next_is_ethics(self):
        footer = self._get_nav_footer("evaluation.md")
        assert footer is not None
        assert "ethics" in footer

    def test_ethics_prev_is_evaluation(self):
        footer = self._get_nav_footer("ethics.md")
        assert footer is not None
        assert "evaluation" in footer

    def test_ethics_next_is_rag(self):
        footer = self._get_nav_footer("ethics.md")
        assert footer is not None
        assert "rag" in footer

    def test_rag_prev_is_ethics(self):
        footer = self._get_nav_footer("rag.md")
        assert footer is not None
        assert "ethics" in footer

    def test_rag_next_is_agents(self):
        footer = self._get_nav_footer("rag.md")
        assert footer is not None
        assert "agents" in footer

    def test_agents_prev_is_rag(self):
        footer = self._get_nav_footer("agents.md")
        assert footer is not None
        assert "rag" in footer

    def test_agents_next_is_safety_alignment(self):
        footer = self._get_nav_footer("agents.md")
        assert footer is not None
        assert "safety-alignment" in footer

    def test_safety_alignment_prev_is_agents(self):
        footer = self._get_nav_footer("safety-alignment.md")
        assert footer is not None
        assert "agents" in footer

    def test_safety_alignment_has_completion_message(self):
        text = read(ADVANCED_DOCS_DIR / "safety-alignment.md")
        assert (
            "Completed" in text
            or "completed" in text
            or "congratulations" in text.lower()
        ), "safety-alignment.md (last doc) should have a completion message"


# ---------------------------------------------------------------------------
# 6. Supplementary Document Navigation Tests
# ---------------------------------------------------------------------------


class TestSupplementaryNavigation:
    """Supplementary numbered docs have valid navigation footers."""

    def test_01_neural_networks_links_to_advanced_home(self):
        text = read(ADVANCED_DOCS_DIR / "01-neural-networks.md")
        assert "Advanced Home" in text or "../README.md" in text

    def test_01_neural_networks_next_is_02(self):
        text = read(ADVANCED_DOCS_DIR / "01-neural-networks.md")
        assert "02-deep-learning-architectures" in text

    def test_02_deep_learning_prev_is_01(self):
        text = read(ADVANCED_DOCS_DIR / "02-deep-learning-architectures.md")
        assert "01-neural-networks" in text

    def test_07_training_next_is_08_evaluation(self):
        text = read(ADVANCED_DOCS_DIR / "07-training-techniques.md")
        assert "08-model-evaluation" in text

    def test_08_model_evaluation_prev_is_07(self):
        text = read(ADVANCED_DOCS_DIR / "08-model-evaluation.md")
        assert "07-training-techniques" in text


# ---------------------------------------------------------------------------
# 7. Code Block Tests
# ---------------------------------------------------------------------------


class TestCodeBlocks:
    """Code blocks in the docs are syntactically well-formed."""

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH + SUPPLEMENTARY_DOCS)
    def test_code_fences_are_balanced(self, doc):
        """Every opening ``` must have a matching closing ```."""
        text = read(ADVANCED_DOCS_DIR / doc)
        fence_lines = [
            ln.strip() for ln in text.splitlines() if ln.strip().startswith("```")
        ]
        assert len(fence_lines) % 2 == 0, (
            "{}: Unbalanced code fences ({} triple-backtick lines, expected even).".format(
                doc, len(fence_lines)
            )
        )

    @pytest.mark.parametrize(
        "doc",
        [
            "neural-networks.md",
            "model-architectures.md",
            "large-language-models.md",
            "training-techniques.md",
            "fine-tuning.md",
            "evaluation.md",
            "rag.md",
            "agents.md",
            "safety-alignment.md",
        ],
    )
    def test_python_code_blocks_have_language_tag(self, doc):
        """Python code blocks should be annotated with ```python."""
        text = read(ADVANCED_DOCS_DIR / doc)
        fence_opens = re.findall(r"^```(\w*)$", text, re.MULTILINE)
        python_blocks = [f for f in fence_opens if f == "python"]
        assert python_blocks, (
            "{}: No ```python code blocks found.".format(doc)
        )


# ---------------------------------------------------------------------------
# 8. Content Completeness and Accuracy Tests
# ---------------------------------------------------------------------------


class TestContentCompleteness:
    """Sections are populated with meaningful content."""

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_learning_objectives_has_bullet_points(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        lo_match = re.search(
            r"Learning Objectives\b.*?(?=^---|\Z)", text, re.DOTALL | re.MULTILINE
        )
        assert lo_match, "{}: Could not locate Learning Objectives section".format(doc)
        lo_text = lo_match.group(0)
        bullets = re.findall(r"^[-*]|\bable to\b", lo_text, re.MULTILINE)
        assert len(bullets) >= 3, (
            "{}: Learning Objectives should list at least 3 objectives, found {}".format(
                doc, len(bullets)
            )
        )

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_prerequisites_has_content(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        prereq_match = re.search(
            r"## Prerequisites\b.*?(?=^---|\Z)", text, re.DOTALL | re.MULTILINE
        )
        if prereq_match:
            prereq_text = prereq_match.group(0)
            assert len(prereq_text.strip()) > 60, (
                "{}: Prerequisites section appears empty or too short".format(doc)
            )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_further_reading_has_resources(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        fr_match = re.search(
            r"Further Reading\b.*?(?=^---|\Z)", text, re.DOTALL | re.MULTILINE
        )
        assert fr_match, "{}: Could not locate Further Reading section".format(doc)
        fr_text = fr_match.group(0)
        links = extract_markdown_links(fr_text)
        assert links, "{}: Further Reading section has no links".format(doc)

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_key_concepts_summary_is_a_table(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        kc_match = re.search(
            r"Key Concepts\b.*?(?=^---|\Z)", text, re.DOTALL | re.MULTILINE
        )
        assert kc_match, "{}: Key Concepts section not found".format(doc)
        kc_text = kc_match.group(0)
        table_rows = [ln for ln in kc_text.splitlines() if "|" in ln]
        assert len(table_rows) >= 3, (
            "{}: Key Concepts Summary should be a table with at least 3 rows, "
            "found {}".format(doc, len(table_rows))
        )

    @pytest.mark.parametrize(
        "doc",
        SUPPLEMENTARY_DOCS + ["deep-learning-architectures.md", "ethics.md", "outline.md"],
    )
    def test_has_last_updated_date(self, doc):
        """Supplementary docs and select main docs should carry a 'Last updated' stamp."""
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "last updated" in text.lower() or "Last updated" in text, (
            "{}: Missing 'Last updated' date stamp".format(doc)
        )

    def test_advanced_readme_has_last_updated_date(self):
        text = read(ADVANCED_README)
        assert "last updated" in text.lower() or "Last updated" in text

    def test_repo_readme_has_last_updated_date(self):
        text = read(REPO_README)
        assert "last updated" in text.lower() or "Last updated" in text

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_doc_has_at_least_three_h2_sections(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        h2s = [title for level, title in get_headings(text) if level == 2]
        assert len(h2s) >= 3, (
            "{}: Expected at least 3 H2 sections, found {}: {}".format(
                doc, len(h2s), h2s
            )
        )


# ---------------------------------------------------------------------------
# 9. Breadcrumb Accuracy Tests
# ---------------------------------------------------------------------------


class TestBreadcrumbAccuracy:
    """Breadcrumbs correctly reference the hierarchy: Home -> Advanced -> Topic."""

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_breadcrumb_links_to_repo_readme(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "../../README.md" in text or "Home" in text, (
            "{}: Breadcrumb should link back to root README.md".format(doc)
        )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_breadcrumb_links_to_advanced_readme(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        assert "../README.md" in text, (
            "{}: Breadcrumb should link to advanced/README.md via ../README.md".format(doc)
        )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_breadcrumb_links_resolve(self, doc):
        path = ADVANCED_DOCS_DIR / doc
        text = read(path)
        breadcrumb_lines = [
            line
            for line in text.splitlines()[:10]
            if "\U0001f4cd" in line or ("Home" in line and "Advanced" in line)
        ]
        assert breadcrumb_lines, (
            "{}: Could not find breadcrumb in first 10 lines".format(doc)
        )
        for line in breadcrumb_lines:
            for link_text, href in extract_markdown_links(line):
                if is_local_file_link(href):
                    resolved = resolve_link(path, href)
                    assert resolved.exists(), (
                        "{}: Breadcrumb link [{}]({}) -> {} does not exist".format(
                            doc, link_text, href, resolved
                        )
                    )


# ---------------------------------------------------------------------------
# 10. Cross-Reference Consistency Tests
# ---------------------------------------------------------------------------


class TestCrossReferences:
    """Prerequisites cross-references and in-text links point to real files."""

    @pytest.mark.parametrize("doc", STRICTLY_STRUCTURED_DOCS)
    def test_prerequisites_links_resolve(self, doc):
        path = ADVANCED_DOCS_DIR / doc
        text = read(path)
        prereq_match = re.search(
            r"## Prerequisites\b.*?(?=^---|\Z)", text, re.DOTALL | re.MULTILINE
        )
        if not prereq_match:
            return
        prereq_text = prereq_match.group(0)
        broken = []
        for link_text, href in extract_markdown_links(prereq_text):
            if not is_local_file_link(href):
                continue
            resolved = resolve_link(path, href)
            if not resolved.exists():
                broken.append("  [{}]({}) -> {}".format(link_text, href, resolved))
        assert not broken, (
            "{}: Broken prerequisite links:\n".format(doc) + "\n".join(broken)
        )

    def test_neural_networks_prereq_references_beginner(self):
        text = read(ADVANCED_DOCS_DIR / "neural-networks.md")
        assert "beginner" in text.lower()

    def test_model_architectures_prereq_references_neural_networks(self):
        text = read(ADVANCED_DOCS_DIR / "model-architectures.md")
        assert "neural-networks" in text or "Neural Networks" in text

    def test_llm_prereq_references_model_architectures(self):
        text = read(ADVANCED_DOCS_DIR / "large-language-models.md")
        assert "model-architectures" in text or "Model Architectures" in text

    def test_fine_tuning_prereq_references_training_techniques(self):
        text = read(ADVANCED_DOCS_DIR / "fine-tuning.md")
        assert "training-techniques" in text or "Training Techniques" in text

    def test_agents_prereq_references_rag(self):
        text = read(ADVANCED_DOCS_DIR / "agents.md")
        assert "rag" in text.lower() or "RAG" in text

    def test_safety_alignment_prereq_references_fine_tuning(self):
        text = read(ADVANCED_DOCS_DIR / "safety-alignment.md")
        assert (
            "fine-tuning" in text
            or "Fine-Tuning" in text
            or "fine_tuning" in text
        )


# ---------------------------------------------------------------------------
# 11. Accessibility / Semantic Structure Tests
# ---------------------------------------------------------------------------


class TestAccessibility:
    """Documents follow accessible, semantically correct heading structures."""

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_first_heading_is_h1(self, doc):
        """The very first heading in a document must be H1 (document title)."""
        text = read(ADVANCED_DOCS_DIR / doc)
        headings = get_headings(text)
        assert headings, "{}: No headings found at all".format(doc)
        first_level = headings[0][0]
        assert first_level == 1, (
            "{}: First heading should be H1, found H{}: '{}'".format(
                doc, first_level, headings[0][1]
            )
        )

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_no_empty_headings(self, doc):
        text = read(ADVANCED_DOCS_DIR / doc)
        empty = re.findall(r"^#{1,6}\s*$", text, re.MULTILINE)
        assert not empty, "{}: Found {} empty heading(s)".format(doc, len(empty))

    @pytest.mark.parametrize("doc", MAIN_LEARNING_PATH)
    def test_has_section_separators(self, doc):
        """Documents should use --- horizontal rules between sections."""
        text = read(ADVANCED_DOCS_DIR / doc)
        separators = re.findall(r"^---\s*$", text, re.MULTILINE)
        assert len(separators) >= 3, (
            "{}: Expected at least 3 horizontal rule separators (---), found {}".format(
                doc, len(separators)
            )
        )


# ---------------------------------------------------------------------------
# 12. Duplicate Content Detection
# ---------------------------------------------------------------------------


class TestDuplicateContent:
    """The supplementary docs provide different content from main path docs."""

    def test_01_neural_networks_is_not_identical_to_neural_networks(self):
        text_main = read(ADVANCED_DOCS_DIR / "neural-networks.md")
        text_supp = read(ADVANCED_DOCS_DIR / "01-neural-networks.md")
        assert text_main != text_supp

    def test_06_ai_ethics_is_not_identical_to_ethics(self):
        text_main = read(ADVANCED_DOCS_DIR / "ethics.md")
        text_supp = read(ADVANCED_DOCS_DIR / "06-ai-ethics.md")
        assert text_main != text_supp

    def test_07_training_is_not_identical_to_training_techniques(self):
        text_main = read(ADVANCED_DOCS_DIR / "training-techniques.md")
        text_supp = read(ADVANCED_DOCS_DIR / "07-training-techniques.md")
        assert text_main != text_supp


# ---------------------------------------------------------------------------
# 13. Outline Document Tests
# ---------------------------------------------------------------------------


class TestOutlineDocument:
    """The outline.md planning document is present and consistent."""

    def test_outline_exists(self):
        assert (ADVANCED_DOCS_DIR / "outline.md").exists()

    def test_outline_has_topic_structure(self):
        text = read(ADVANCED_DOCS_DIR / "outline.md")
        assert "## Topic 1" in text or "Topic 1:" in text

    def test_outline_covers_neural_networks(self):
        text = read(ADVANCED_DOCS_DIR / "outline.md")
        assert "Neural Networks" in text

    def test_outline_covers_deep_learning_architectures(self):
        text = read(ADVANCED_DOCS_DIR / "outline.md")
        assert "Deep Learning Architectures" in text

    def test_outline_covers_ai_ethics(self):
        text = read(ADVANCED_DOCS_DIR / "outline.md")
        assert "AI Ethics" in text or "Ethics" in text


# ---------------------------------------------------------------------------
# 14. Beginner Section Integration Tests
# ---------------------------------------------------------------------------


class TestBeginnerSectionIntegration:
    """The beginner section exists and is properly referenced from advanced docs."""

    def test_beginner_readme_exists(self):
        assert BEGINNER_README.exists()

    def test_beginner_what_is_ai_exists(self):
        what_is_ai = BEGINNER_DIR / "docs" / "what-is-ai.md"
        assert what_is_ai.exists()

    def test_beginner_how_to_use_ai_exists(self):
        how_to = BEGINNER_DIR / "docs" / "how-to-use-ai.md"
        assert how_to.exists()

    def test_advanced_section_references_beginner_for_newcomers(self):
        text = read(ADVANCED_README)
        assert "Beginner" in text

    def test_neural_networks_links_to_beginner_what_is_ai(self):
        text = read(ADVANCED_DOCS_DIR / "neural-networks.md")
        assert "what-is-ai" in text or "what is ai" in text.lower()

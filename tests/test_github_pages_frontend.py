from pathlib import Path


def test_github_pages_assets_exist():
    base = Path("docs/github-pages")
    for name in ["index.html", "styles.css", "app.js", "README.md"]:
        assert (base / name).exists(), f"Missing {name}"


def test_frontend_has_no_required_api_dependencies():
    """Verify the frontend works without API calls (external links like GitHub are OK)."""
    js = Path("docs/github-pages/app.js").read_text()
    # The JS scoring engine must not require any API calls
    assert "fetch(" not in js
    assert "XMLHttpRequest" not in js


def test_frontend_implements_bucket_rules():
    js = Path("docs/github-pages/app.js").read_text()
    assert "match >= 75 && gap <= 25" in js
    assert "match >= 50 || (gap >= 26 && gap <= 55)" in js


def test_frontend_has_fashion_exit_persona_with_three_fields():
    js = Path("docs/github-pages/app.js").read_text()
    assert "Nia (fashion -> new field)" in js
    assert "Project Coordination" in js
    assert "Operations" in js
    assert "Data Analysis" in js


def test_frontend_exposes_persona_qa_matrix_markup():
    html = Path("docs/github-pages/index.html").read_text()
    assert "Persona QA Matrix" in html or "Transition Summary" in html
    assert 'id="qa-table"' in html


def test_frontend_has_multiple_personas():
    js = Path("docs/github-pages/app.js").read_text()
    assert "Nia" in js
    assert "Marcus" in js
    assert "Priya" in js


def test_frontend_has_empty_states():
    html = Path("docs/github-pages/index.html").read_text()
    assert "empty-ready" in html
    assert "empty-trainable" in html
    assert "empty-long" in html


def test_frontend_has_skill_level_labels():
    js = Path("docs/github-pages/app.js").read_text()
    assert "levelLabels" in js
    assert "None" in js
    assert "Basic" in js
    assert "Intermediate" in js
    assert "Expert" in js


def test_frontend_has_open_graph_meta():
    html = Path("docs/github-pages/index.html").read_text()
    assert 'property="og:title"' in html
    assert 'property="og:description"' in html


def test_frontend_live_updates_on_slider_change():
    """Verify the JS calls runRecommendations on slider input events."""
    js = Path("docs/github-pages/app.js").read_text()
    assert "runRecommendations()" in js
    # Should be called in the slider input event handler
    assert 'addEventListener("input"' in js

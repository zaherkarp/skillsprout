from pathlib import Path


def test_github_pages_assets_exist():
    base = Path("docs/github-pages")
    for name in ["index.html", "styles.css", "app.js", "README.md"]:
        assert (base / name).exists(), f"Missing {name}"


def test_frontend_has_no_external_hosting_dependencies():
    html = Path("docs/github-pages/index.html").read_text()
    assert "https://" not in html
    assert "http://" not in html


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
    assert "Persona QA matrix (three fields)" in html
    assert 'id="qa-table"' in html

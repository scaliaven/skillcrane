"""The docs must not lie about how many tests there are.

Six files carry hand-maintained test counts -- README, CLAUDE.md, DEVLOG.md,
BENCHMARKS.md, environment.yml and the project page -- and all six had drifted,
one of them by four commits and 114 tests. Nothing noticed, because nothing was
looking: every other test in this repo asserts on the arm, and no one re-reads a
number in a table.

Ground truth is what pytest *collects*, gathered in a subprocess so it is the
same answer however this file was invoked (running one test module would
otherwise make the session look tiny). Collection cannot know how many tests will
skip -- these skip at runtime, from `pytest.skip()` inside the benchmark tests --
so the rule is that the numbers a doc states must **sum** to what is collected:
"230 passed, 17 skipped" is 247, and so is "39 (+17 skipped)" summing to 56 on
one row. That has the useful side effect of making the docs machine-independent,
since installing robosuite changes which tests pass but not how many exist.

A number that is deliberately frozen -- the two benchmark-installed worlds, last
measured when the adapters landed -- is exempt, and the exemption *is* the
disclosure: a paragraph is skipped when it says so in words a reader also sees
(see STALE_OK). There is no silent way to opt out.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Everything that quotes a test count. test_every_doc_that_counts_is_listed
#: fails if a new one appears and is not added here.
DOCS = ("README.md", "CLAUDE.md", "DEVLOG.md", "BENCHMARKS.md",
        "environment.yml", "docs/index.html")

#: A *line* containing any of these carries a number the docs deliberately keep
#: as history rather than fact. Each is a phrase the reader sees too, so a frozen
#: number cannot be hidden from a human and exempted from the test at the same
#: time. Line-level and not paragraph-level on purpose: a paragraph exemption let
#: one disclaimer cover a whole markdown table, and the live row in it went
#: unchecked -- which is how CLAUDE.md's own total escaped the first version of
#: this test. The disclosure has to sit where the number does.
STALE_OK = ("re-measured", "measured at", "†")

# "230 passed, 17 skipped" / "230 passed / 17 skipped" -- the shape pytest itself
# prints, which is why the docs are written in it.
TOTAL_RE = re.compile(r"(\d+)\s*(?:passed|tests)\s*[,/]\s*(\d+)\s*skipped")
# A row that names a test module: sum every integer after the filename.
FILE_RE = re.compile(r"`(test_\w+\.py)`([^|\n]*\|[^|\n]*)")
# The project page's milestone table, which names milestones rather than files.
HTML_ROW_RE = re.compile(r'class="mono id">(M\d+)</td>'
                         r'.*?<td class="num[^"]*">([^<]*)</td>')
MILESTONES = {"M1": "test_m1_scene_kin.py", "M2": "test_m2_tracking.py",
              "M3": "test_m3_grasp.py", "M4": "test_m4_rules.py",
              "M5": "test_m5_input.py", "M6": "test_m6_render.py",
              "M7": "test_m7_record.py", "M8": "test_benchmarks.py",
              "M9": "test_m9_policy.py"}
#: The rows under the milestones, which the page labels rather than numbers.
OTHER_ROWS = {"Hard rule": "test_no_pygame.py", "Docs": "test_docs.py"}
#: Every module the docs account for. The suite total is only ours to check when
#: all of them collected: `test_m7_record.py` importorskips pyarrow at module
#: level, so on a machine without it those 23 tests do not exist and a documented
#: total of 258 is right about this repo and wrong about this machine.
ACCOUNTED = frozenset(MILESTONES.values()) | frozenset(OTHER_ROWS.values())


@pytest.fixture(scope="module")
def collected():
    """{filename: tests collected}, plus "" for the total, from a real pytest.

    A subprocess rather than `request.session`, which only holds what this
    invocation collected -- `pytest tests/test_docs.py` would otherwise report
    that the suite is four tests long and the docs are wrong.
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                        "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    counts = {}
    for line in r.stdout.splitlines():
        if line.startswith("tests/") and "::" in line:
            counts[Path(line.split("::")[0]).name] = \
                counts.get(Path(line.split("::")[0]).name, 0) + 1
    assert counts, f"collected nothing:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    counts[""] = sum(counts.values())
    return counts


def documented(doc):
    """[(line, text, stated numbers, filename or "" for the whole suite)].

    Only lines claiming a *current* number: one that discloses it is frozen is
    skipped, and skipped alone, so the live row of a table with a historical row
    beside it is still checked.
    """
    out = []
    for n, line in enumerate((ROOT / doc).read_text().splitlines(), 1):
        if any(k in line for k in STALE_OK):
            continue
        named = FILE_RE.search(line)
        if named:
            nums = [int(x) for x in re.findall(r"\d+", named.group(2))]
            if nums:
                out.append((n, line.strip(), nums, named.group(1)))
            continue
        for m in TOTAL_RE.finditer(line):
            out.append((n, line.strip(), [int(m.group(1)), int(m.group(2))], ""))
    return out


# --- the checks --------------------------------------------------------------

@pytest.mark.parametrize("doc", DOCS)
def test_documented_counts_match_the_suite(doc, collected):
    """Every current test count in every doc, against what pytest collects."""
    whole_suite = ACCOUNTED <= set(collected)
    wrong = []
    for line, text, nums, name in documented(doc):
        if name == "" and not whole_suite:
            continue        # an optional dep is missing; the total is not ours
        want = collected.get(name)
        if want is None:
            continue        # module-level importorskip: nothing collected here
        if sum(nums) != want:
            wrong.append(f"  {doc}:{line}  says {nums} (sum {sum(nums)}), "
                         f"pytest collects {want} for {name or 'the suite'}\n"
                         f"      {text[:100]}")
    assert not wrong, "documented test counts have drifted:\n" + "\n".join(wrong)


def test_the_project_page_milestone_table_matches(collected):
    """docs/index.html names milestones, not files, so it needs its own reading.

    It is also the doc that drifted worst -- it read "116 tests passing" while
    the suite was at 230 -- because it is the one nobody opens while working.
    """
    html = (ROOT / "docs/index.html").read_text()
    rows = dict(HTML_ROW_RE.findall(html))
    assert rows, "no milestone table found on the page"
    for label in OTHER_ROWS:
        m = re.search(label + r'.*?<td class="num[^"]*">([^<]*)</td>', html)
        assert m, f"no {label!r} row found on the page"
        rows[label] = m.group(1)

    wrong = []
    for key, cell in rows.items():
        name = MILESTONES.get(key) or OTHER_ROWS[key]
        want = collected.get(name)
        if want is None:
            continue
        nums = [int(x) for x in re.findall(r"\d+", cell)]
        if sum(nums) != want:
            wrong.append(f"  {key} ({name}): page says {cell.strip()!r} "
                         f"(sum {sum(nums)}), pytest collects {want}")
    assert not wrong, "the project page's milestone table has drifted:\n" + \
                      "\n".join(wrong)
    assert set(rows) - set(OTHER_ROWS) == set(MILESTONES), \
        "the page's milestone rows and MILESTONES disagree about which exist"


def test_every_doc_that_counts_is_listed():
    """A new doc quoting test counts must join DOCS, not drift unwatched."""
    missing = []
    for path in list(ROOT.glob("*.md")) + list(ROOT.glob("*.yml")) + \
            list(ROOT.glob("docs/*.html")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in DOCS:
            continue
        if TOTAL_RE.search(path.read_text()):
            missing.append(rel)
    assert not missing, f"these quote test counts but are not in DOCS: {missing}"


def test_the_docs_account_for_every_test_module(collected):
    """A new tests/ module has to appear in the milestone tables.

    The counts staying right is only half of it: a module nobody documented
    drifts by being invisible rather than by being wrong.
    """
    found = {f for f in collected if f}
    assert found == ACCOUNTED, (
        f"undocumented modules: {sorted(found - ACCOUNTED)}; "
        f"documented but not collected: {sorted(ACCOUNTED - found)}")


def test_every_listed_doc_actually_states_a_count():
    """The other direction: a doc that stopped counting should leave DOCS.

    Otherwise DOCS slowly fills with files this test is silently doing nothing
    for, and the coverage it advertises stops being real.
    """
    silent = [d for d in DOCS if not documented(d)]
    assert not silent, f"these are in DOCS but state no current count: {silent}"


# --- teeth -------------------------------------------------------------------

def test_drift_is_actually_detected(tmp_path, monkeypatch):
    """The check has to be able to fail, or it is decoration.

    Points the reader at a doc whose numbers are wrong on purpose and asserts
    both that it is caught and that the message names the file and the line.
    """
    doc = tmp_path / "FAKE.md"
    doc.write_text("Totals: **9999 passed / 1 skipped**\n\n"
                   "| M1 | scene | `test_m1_scene_kin.py` | 4321 passed |\n")
    monkeypatch.setattr("tests.test_docs.ROOT", tmp_path, raising=False)
    import tests.test_docs as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    found = mod.documented("FAKE.md")
    assert [n for _, _, n, _ in found] == [[9999, 1], [4321]]
    assert [f for _, _, _, f in found] == ["", "test_m1_scene_kin.py"]

    with pytest.raises(AssertionError, match="drifted"):
        mod.test_documented_counts_match_the_suite(
            "FAKE.md", {"": 247, "test_m1_scene_kin.py": 15,
                        **{v: 1 for v in mod.MILESTONES.values()},
                        "test_no_pygame.py": 1, "test_docs.py": 1})


def test_a_frozen_number_is_exempt_only_when_it_says_so(tmp_path, monkeypatch):
    """The exemption is the disclosure -- and it cannot be applied invisibly."""
    import tests.test_docs as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    (tmp_path / "A.md").write_text("It last read 125 passed / 1 skipped.\n")
    assert mod.documented("A.md"), "an unlabelled stale number must be checked"

    (tmp_path / "B.md").write_text(
        "It last read 125 passed / 1 skipped, and has not been re-measured.\n")
    assert not mod.documented("B.md"), "a labelled one must be exempt"

    # And the exemption must not spill onto its neighbours: one frozen row in a
    # table cannot excuse the live row under it.
    (tmp_path / "C.md").write_text(
        "| core | 241 passed, 17 skipped |\n"
        "| with robosuite | 125 passed, 1 skipped, not re-measured |\n")
    assert [n for _, _, n, _ in mod.documented("C.md")] == [[241, 17]]

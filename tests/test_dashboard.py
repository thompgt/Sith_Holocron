"""Keep the Grafana dashboard and the metric definitions from drifting apart.

A dashboard panel that queries a renamed or deleted metric does not fail; it
renders an empty graph, which looks exactly like "no traffic yet". That is the
worst possible failure mode for observability, and nothing else in this repo
would catch it -- the JSON is only ever read by Grafana at runtime.
"""

import json
import re
from pathlib import Path

import pytest
from prometheus_client import REGISTRY

import src.observability.metrics  # noqa: F401  (registers the families)

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards"

#: Emitted by prometheus-fastapi-instrumentator rather than declared in
#: src/observability/metrics.py, so the registry does not know about them here.
INSTRUMENTATOR_FAMILIES = {"http_requests", "http_request_duration_seconds"}

#: The uid provisioned in monitoring/grafana/provisioning/datasources/prometheus.yml.
DATASOURCE_UID = "holocron-prometheus"


def dashboard_files():
    return sorted(DASHBOARD_DIR.glob("*.json"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def walk_strings(node, key):
    """Yield every value stored under `key` anywhere in the document."""
    if isinstance(node, dict):
        value = node.get(key)
        if isinstance(value, str):
            yield value
        for child in node.values():
            yield from walk_strings(child, key)
    elif isinstance(node, list):
        for child in node:
            yield from walk_strings(child, key)


def family_of(series_name):
    """Strip the suffix prometheus_client appends to reach the family name."""
    for suffix in ("_bucket", "_sum", "_count", "_total"):
        if series_name.endswith(suffix):
            return series_name[: -len(suffix)]
    return series_name


def referenced_families(dashboard):
    names = set()
    for expr in [*walk_strings(dashboard, "expr"), *walk_strings(dashboard, "query")]:
        names |= set(re.findall(r"\b(?:holocron|http)_[a-z_]+\b", expr))
    return {family_of(n) for n in names}


def declared_families():
    return {metric.name for metric in REGISTRY.collect()}


def test_there_is_a_dashboard():
    """monitoring/docker-compose.yml mounts this directory into Grafana. It was
    absent from git, so the bind mount resolved to an empty directory and the
    provisioned dashboard folder came up with nothing in it."""
    assert dashboard_files(), f"no dashboard JSON in {DASHBOARD_DIR}"


@pytest.mark.parametrize("path", dashboard_files(), ids=lambda p: p.name)
def test_every_queried_metric_exists(path):
    unknown = sorted(
        family
        for family in referenced_families(load(path))
        if family not in declared_families() and family not in INSTRUMENTATOR_FAMILIES
    )
    assert not unknown, f"{path.name} queries metrics that are not defined: {unknown}"


@pytest.mark.parametrize("path", dashboard_files(), ids=lambda p: p.name)
def test_every_declared_metric_is_shown(path):
    """A metric nobody plots is a metric nobody reads."""
    shown = referenced_families(load(path))
    missing = sorted(f for f in declared_families() if f.startswith("holocron_") and f not in shown)
    assert not missing, f"{path.name} does not plot: {missing}"


@pytest.mark.parametrize("path", dashboard_files(), ids=lambda p: p.name)
def test_panels_use_the_provisioned_datasource(path):
    dashboard = load(path)
    uids = {
        panel.get("datasource", {}).get("uid")
        for panel in dashboard["panels"]
        if panel["type"] != "row"
    }
    assert uids == {DATASOURCE_UID}, f"{path.name} references datasources {uids}"


@pytest.mark.parametrize("path", dashboard_files(), ids=lambda p: p.name)
def test_panel_ids_are_unique(path):
    """Grafana silently drops the duplicate rather than reporting a conflict."""
    ids = [panel["id"] for panel in load(path)["panels"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("path", dashboard_files(), ids=lambda p: p.name)
def test_dashboard_has_a_stable_uid(path):
    """Provisioning re-imports on every restart; without a uid Grafana mints a
    new one each time and old links rot."""
    dashboard = load(path)
    assert dashboard.get("uid")
    assert dashboard.get("title")

"""Shared fixtures for Heartwood stress tests.

Creates two real Supabase test users (Alice and Bob) for multi-tenant
isolation testing.  All test data is cleaned up on teardown.

Required env vars:
    SUPABASE_URL          — project URL
    SUPABASE_KEY          — anon/public key (for client-side auth)
    SUPABASE_SERVICE_KEY  — service_role key (for admin cleanup only)

If any are missing, all tests in this suite are skipped.
"""

from __future__ import annotations

import os
import sys
import uuid
import pytest

# Allow bare imports from cerebro/
_cerebro_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(_cerebro_dir))

# Auto-load cerebro/.env so tests pick up credentials
from dotenv import load_dotenv
load_dotenv(os.path.join(_cerebro_dir, '.env'))

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") or os.environ.get("SUPABASE_SERVICE_KEY", "")

needs_supabase = pytest.mark.skipif(
    not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY]),
    reason="SUPABASE_URL, SUPABASE_KEY, and SUPABASE_SERVICE_ROLE_KEY required",
)


def _create_test_user(email: str, password: str):
    """Sign up a test user via Supabase Auth, return (user_id, access_token)."""
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = client.auth.sign_up({"email": email, "password": password})
    if not resp.user or not resp.session:
        raise RuntimeError(f"Failed to create test user {email}: {resp}")
    return resp.user.id, resp.session.access_token


def _delete_test_user(user_id: str):
    """Delete a test user via service_role (admin) client."""
    from supabase import create_client
    admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    # admin.auth.admin is the GoTrue admin API
    admin.auth.admin.delete_user(user_id)


def _make_backend(user_id: str, access_token: str):
    from storage_supabase import SupabaseBackend
    return SupabaseBackend(SUPABASE_URL, SUPABASE_KEY, access_token, user_id)


def _cleanup_user_data(user_id: str):
    """Wipe all data for a user via service_role (bypasses RLS)."""
    from supabase import create_client
    admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    for table in ['notes', 'connections', 'canvases', 'ontology',
                  'user_graph_state', 'cli_tokens', 'telegram_links',
                  'telegram_link_tokens']:
        try:
            admin.table(table).delete().eq('user_id', str(user_id)).execute()
        except Exception:
            pass  # table may not exist yet


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def alice():
    """Test user Alice — creates on first use, deletes on teardown."""
    tag = uuid.uuid4().hex[:8]
    email = f"alice-test-{tag}@cerebro-test.local"
    password = f"TestPass!{tag}"
    user_id, token = _create_test_user(email, password)
    backend = _make_backend(user_id, token)
    yield {"user_id": user_id, "token": token, "email": email,
           "password": password, "backend": backend}
    _cleanup_user_data(user_id)
    _delete_test_user(user_id)


@pytest.fixture(scope="session")
def bob():
    """Test user Bob — creates on first use, deletes on teardown."""
    tag = uuid.uuid4().hex[:8]
    email = f"bob-test-{tag}@cerebro-test.local"
    password = f"TestPass!{tag}"
    user_id, token = _create_test_user(email, password)
    backend = _make_backend(user_id, token)
    yield {"user_id": user_id, "token": token, "email": email,
           "password": password, "backend": backend}
    _cleanup_user_data(user_id)
    _delete_test_user(user_id)


@pytest.fixture(scope="session")
def server_url():
    """URL of a running Heartwood server, or skip."""
    url = os.environ.get("CEREBRO_SERVER_URL", "")
    if not url:
        pytest.skip("CEREBRO_SERVER_URL required for HTTP-level tests")
    return url

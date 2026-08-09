"""
One-time setup script: creates the Databricks secret scope(s) and stores the
Lakebase URL. The NWS + geocoding APIs need no credentials, so this app only
requires the single `database/lakebase-url` secret.

Run locally (with the Databricks CLI configured) or from a notebook with
`%sh python setup_secrets.py` - never commit the resulting secret value.

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

# w.secrets.create_scope(scope="database")
w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: ")
)

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)
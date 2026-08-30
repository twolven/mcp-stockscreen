# Changelog
## 2.0.0
- Validated `Host` and `Origin` headers on the Streamable HTTP transport so loopback deployments reject DNS-rebinding requests, configurable through `MCP_HOST_ORIGIN_PROTECTION`, `MCP_ALLOWED_HOSTS`, and `MCP_ALLOWED_ORIGINS`.
- Migrated to typed FastMCP, documented Yahoo EquityQuery discovery, normalized current news, rejected missing metrics, retained rejection reasons, and made persistence atomic and traversal-safe.
- Restored all legacy criteria names/paths, reject unknown criteria, corrected custom-query size/sort/region/category semantics, normalize persisted results, and added captured fixtures plus branch-coverage gates.
- Restored the remaining market-cap, margin, leverage, price/book, and expiration-day bounds identified by follow-up verification.
- Added a non-root Docker/Compose deployment with localhost-bound Streamable HTTP, persistent datastore volume, health checks, and an end-to-end container contract gate.

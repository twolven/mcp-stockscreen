# Plan de migration — Stockscreen monolithe → package modulaire

> **Méthode** : TDD strict — pour chaque étape : écrire les tests → migrer le code → lancer les tests
> **Source** : `stockscreen.py` (1583 lignes)
> **Cible** : package `stockscreen/` avec FastMCP, Pydantic, async réel
> **Règle** : l'ancien `stockscreen.py` n'est supprimé qu'à la toute fin, quand tout est migré et testé

---

## Phase 0 — Préparation

### 0.1 Mettre à jour les dépendances
> **Environnement virtuel** : `venv/` à la racine du projet. Toujours activer avec `source venv/bin/activate` avant d'installer ou de lancer quoi que ce soit.

- [x] Mettre à jour `requirements.txt` :
  - `mcp>=1.6` (inclut FastMCP)
  - `yfinance>=1.2.0`
  - `pydantic>=2.0`
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
  - Supprimer `asyncio` (stdlib, pas besoin de le lister)
- [x] Mettre à jour `pyproject.toml` avec les dépendances et le point d'entrée
- [x] `pip install -r requirements.txt` — vérifier que tout s'installe
- [x] Lancer les tests existants (`pytest`) — s'assurer qu'ils passent encore (ou noter les cassés par la montée de version yfinance)

### 0.2 Créer la structure du package
- [x] Créer l'arborescence :
  ```
  stockscreen/
  ├── __init__.py
  ├── server.py
  ├── config.py
  ├── exceptions.py
  ├── providers/
  │   ├── __init__.py
  │   └── yahoo.py
  ├── models/
  │   ├── __init__.py
  │   └── schemas.py
  ├── services/
  │   ├── __init__.py
  │   ├── screener.py
  │   ├── news.py
  │   └── watchlist.py
  └── store/
      ├── __init__.py
      └── data_store.py
  ```
- [x] Créer les `__init__.py` vides (ou avec imports minimaux)

---

## Phase 1 — Fondations (modules sans dépendance métier)

### 1.1 `stockscreen/config.py` — Configuration et logging
**Contenu** : paths, constantes, setup logging, migration legacy
**Source** : lignes 20-60 de `stockscreen.py`

- [x] Écrire `tests/test_config.py`
  - Test : `DEFAULT_DATA_PATH` utilise la variable d'env si définie
  - Test : `DEFAULT_LOG_PATH` est relatif au script
  - Test : `migrate_legacy_data()` ne migre pas si env var définie
- [x] Migrer le code dans `stockscreen/config.py`
- [x] Lancer `pytest tests/test_config.py` — vert (8/8)

### 1.2 `stockscreen/exceptions.py` — Hiérarchie d'erreurs
**Contenu** : `StockscreenError`, `ValidationError`, `APIError`
**Source** : lignes 62-69 de `stockscreen.py`

- [x] Écrire `tests/test_exceptions.py`
  - Test : hiérarchie d'héritage
- [x] Migrer le code dans `stockscreen/exceptions.py`
- [x] Lancer `pytest tests/test_exceptions.py` — vert (7/7)

### 1.3 `stockscreen/models/schemas.py` — Modèles Pydantic
**Contenu** : modèles de validation (remplace `validate_watchlist_name`, `validate_stock_symbols`), modèles de critères, `StockscreenJSONEncoder`
**Source** : lignes 232-369 de `stockscreen.py`

- [x] Écrire `tests/test_schemas.py`
  - Tests validation watchlist name (repris des tests existants, adaptés à Pydantic)
  - Tests validation stock symbols
  - Tests JSON encoder (Timestamp, NaT, Period, date, numpy, NaN)
- [x] Créer `stockscreen/models/schemas.py`
- [x] Lancer `pytest tests/test_schemas.py` — vert (28/28)

---

## Phase 2 — Couche données

### 2.1 `stockscreen/store/data_store.py` — Persistance
**Contenu** : `ScreenerDataStore` (une seule définition !), `DefaultSymbols`
**Source** : lignes 71-204, 372-427 de `stockscreen.py`

- [x] Écrire `tests/test_data_store.py`
  - Tests CRUD watchlists (repris des tests existants)
  - Tests CRUD screening results
  - Tests création automatique des répertoires
  - Tests DefaultSymbols : filtrage par catégorie, cache
- [x] Migrer le code dans `stockscreen/store/data_store.py`
  - Fusionner les deux définitions de `ScreenerDataStore`
  - Importer exceptions depuis `stockscreen.exceptions`
  - Importer encoder depuis `stockscreen.models.schemas`
- [x] Lancer `pytest tests/test_data_store.py` — vert (21/21)

---

## Phase 3 — Provider (accès données externes)

### 3.1 `stockscreen/providers/yahoo.py` — Wrapper yfinance async
**Contenu** : classe `YahooProvider` qui encapsule tous les appels `yf.Ticker`
**Source** : tous les appels `yf.Ticker(symbol)` dispersés dans `stockscreen.py`

- [x] Écrire `tests/test_yahoo_provider.py`
  - Test : `get_ticker_info(symbol)` retourne un dict
  - Test : `get_history(symbol, period)` retourne un DataFrame
  - Test : `get_option_chain(symbol, expiry)` retourne calls/puts
  - Test : `get_news(symbol)` retourne une liste
  - Test : `get_earnings_dates(symbol)` retourne le bon format
  - Test : les appels passent par `run_in_executor` (vrai async)
  - Test : retry avec backoff exponentiel sur erreur réseau
- [x] Créer `stockscreen/providers/yahoo.py`
  - `YahooProvider` avec méthodes async
  - `run_in_executor` pour chaque appel yfinance
  - Décorateur `_retry` intégré ici (pas sur le dispatcher MCP)
- [x] Lancer `pytest tests/test_yahoo_provider.py` — vert (16/16)

---

## Phase 4 — Services (logique métier)

### 4.1 `stockscreen/services/news.py` — Service news
**Contenu** : récupération, catégorisation et filtrage des news
**Source** : `get_news_data()`, `run_news_screen()` de `stockscreen.py`

- [x] Écrire `tests/test_news_service.py`
  - Test : catégorisation des news (management, key_events, recent_news)
  - Test : filtrage par keywords, exclude_keywords, require_all
  - Test : filtrage par date (min_days, max_days)
  - Test : gestion des erreurs (pas de news, erreur API)
- [x] Créer `stockscreen/services/news.py`
  - Classe `NewsService` qui prend un `YahooProvider` en injection
  - Aucun import de `yfinance` direct
- [x] Lancer `pytest tests/test_news_service.py` — vert (12/12)

### 4.2 `stockscreen/services/watchlist.py` — Service watchlists
**Contenu** : CRUD watchlists avec validation
**Source** : branche `manage_watchlist` du `call_tool()` de `stockscreen.py`

- [x] Écrire `tests/test_watchlist_service.py`
  - Test : create, get, update, delete
  - Test : validation du nom (via Pydantic)
  - Test : validation des symboles (via Pydantic)
  - Test : erreurs (watchlist not found, etc.)
- [x] Créer `stockscreen/services/watchlist.py`
  - Classe `WatchlistService` qui prend un `DataStore` en injection
- [x] Lancer `pytest tests/test_watchlist_service.py` — vert (13/13)

### 4.3 `stockscreen/services/screener.py` — Service screening unifié
**Contenu** : screening technique, fondamental, options, custom — **unifié, sans duplication**
**Source** : toutes les fonctions `run_*_screen` et `run_single_*_screen` de `stockscreen.py`

- [ ] Écrire `tests/test_screener_service.py`
  - Tests screening technique (prix, volume, RSI, SMA, ATR)
  - Tests screening fondamental (market cap, PE, dividendes, revenue growth, ETF)
  - Tests screening options (IV, volume, put/call ratio, spreads, earnings)
  - Tests screening custom (combinaison technique + fondamental + options + news)
  - Tests : symboles depuis watchlist, depuis critères, ou défaut
  - Tests : résultats rejetés avec raisons
  - Tests : erreur sur un symbole n'arrête pas les autres
- [ ] Créer `stockscreen/services/screener.py`
  - Classe `ScreenerService` avec injection de `YahooProvider`, `DataStore`, `NewsService`
  - UNE seule méthode `_screen_single()` par type (plus de duplication `run_*` / `run_single_*`)
  - Méthode publique `run(screen_type, criteria, watchlist_name)` qui boucle sur les symboles
- [ ] Lancer `pytest tests/test_screener_service.py` — vert

---

## Phase 5 — Serveur FastMCP

### 5.1 `stockscreen/server.py` — Point d'entrée FastMCP
**Contenu** : déclaration des tools, câblage vers les services, `main()`
**Source** : `list_tools()`, `call_tool()`, `main()` de `stockscreen.py`

- [ ] Écrire `tests/test_server.py`
  - Test : le serveur FastMCP s'initialise
  - Test : les tools sont déclarés (run_stock_screen, get_stock_news, manage_watchlist, get_screening_result)
  - Test : appel d'un tool route vers le bon service
  - Test : erreurs de validation retournent une réponse propre
- [ ] Créer `stockscreen/server.py`
  - `FastMCP("stockscreen")`
  - Un `@mcp.tool()` par outil
  - Instanciation des services avec injection des dépendances
  - `mcp.run()` dans `main()`
- [ ] Lancer `pytest tests/test_server.py` — vert

### 5.2 `stockscreen/__init__.py` — Exports du package
- [ ] Configurer les imports publics dans `__init__.py`
- [ ] Mettre à jour `pyproject.toml` avec le point d'entrée : `stockscreen.server:main`

---

## Phase 6 — Intégration et nettoyage

### 6.1 Tests d'intégration end-to-end
- [ ] Écrire `tests/test_integration.py`
  - Test : un screening technique complet (mock yfinance au niveau provider)
  - Test : cycle watchlist create → screen → get result
  - Test : screening custom multi-critères
- [ ] Lancer `pytest tests/test_integration.py` — vert

### 6.2 Lancer toute la suite de tests
- [ ] `pytest` — tous les tests passent

### 6.3 Mettre à jour la configuration
- [ ] Mettre à jour `.mcp.json` si besoin (nouveau point d'entrée)
- [ ] Mettre à jour `CLAUDE.md` avec la nouvelle architecture

### 6.4 Nettoyage
- [ ] Supprimer `stockscreen.py` (l'ancien monolithe)
- [ ] Supprimer `tests/test_stockscreen.py` (remplacé par les nouveaux tests)
- [ ] Supprimer les anciens `conftest.py` fixtures si devenues inutiles
- [ ] `pytest` — dernière vérification, tout est vert

---

## Résumé de l'architecture cible

```
stockscreen/
├── __init__.py            # Exports publics
├── server.py              # FastMCP tools (passe-plat vers services)
├── config.py              # Paths, logging, constantes
├── exceptions.py          # StockscreenError, ValidationError, APIError
├── providers/
│   └── yahoo.py           # YahooProvider (seul fichier qui importe yfinance)
├── models/
│   └── schemas.py         # Pydantic models + JSON encoder
├── services/
│   ├── screener.py        # ScreenerService (technique, fondamental, options, custom)
│   ├── news.py            # NewsService
│   └── watchlist.py       # WatchlistService
└── store/
    └── data_store.py      # ScreenerDataStore + DefaultSymbols
```

### Principes
| Règle | Raison |
|---|---|
| `server.py` n'importe jamais `yfinance` | Découplage MCP / données |
| `services/` n'importe jamais `mcp` | Testable sans serveur MCP |
| `providers/` est le seul à faire du I/O réseau | Un seul endroit pour retry, rate-limit, executor |
| Pydantic pour la validation | Auto-généré par FastMCP, plus de schemas JSON manuels |
| Injection de dépendances | Tests unitaires avec mock du provider |

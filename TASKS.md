# Tâches d'implémentation — Façade & Palmarès Dividendes

> Référence : [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
> Méthodologie : TDD — les tests sont écrits **avant** le code de production.
> Convention : cocher `[x]` dès qu'une tâche est terminée et les tests passent au vert.

---

## Étape 1 — EuronextProvider (SF-1)

### Tests (`tests/test_euronext_provider.py`)

- [x] **1.1** Fixture : réponse JSON Euronext ISIN→ticker (`XPAR`, `XETR`, `XLON`, `XAMS`, MIC inconnu)
- [x] **1.2** Fixture : réponse JSON Euronext ticker→ISIN (search)
- [x] **1.3** `resolve_ticker` : retourne un `EuronextRecord` avec `yahoo_ticker = "TTE.PA"` pour ISIN XPAR
- [x] **1.4** `resolve_ticker` : suffixe `.DE` pour MIC `XETR`
- [x] **1.5** `resolve_ticker` : suffixe vide pour MIC inconnu
- [x] **1.6** `resolve_ticker` : retourne `None` si ISIN non trouvé (réponse vide)
- [x] **1.7** `resolve_isin` : retourne un `EuronextRecord` depuis ticker `"TTE"`
- [x] **1.8** `resolve_isin` : normalise le ticker — `"TTE.PA"` → appel avec `"TTE"`
- [x] **1.9** `resolve_isin` : retourne `None` si ticker non trouvé
- [x] **1.10** Cache : `resolve_ticker` sur ISIN déjà en cache ne fait pas d'appel HTTP
- [x] **1.11** Cache : `resolve_isin` sur ticker déjà en cache ne fait pas d'appel HTTP
- [x] **1.12** Cache : cache expiré (TTL dépassé) déclenche un nouvel appel HTTP
- [x] **1.13** Cache : fichier partagé entre `resolve_ticker` et `resolve_isin` pour le même record
- [x] **1.14** Cache : erreur réseau → stale fallback si cache disponible
- [x] **1.15** Cache : erreur réseau sans cache → retourne `None` (pas d'exception)
- [x] **1.16** `invalidate_cache` : supprime le fichier cache (ISIN ou ticker)
- [x] **1.17** `invalidate_cache` : ne lève pas d'exception si le fichier n'existe pas
- [x] **1.18** Tous les appels HTTP s'exécutent dans `run_in_executor` (non bloquant)
- [x] **1.19** `EuronextRecord` : `yahoo_ticker` construit correctement pour chaque MIC de la table
- [x] **1.20** `EuronextRecord` : `cached_at` est un ISO timestamp valide

### Code de production (`stockscreen/providers/euronext.py`)

- [x] **1.21** Dataclass `EuronextRecord` avec tous les champs
- [x] **1.22** Table `_MIC_TO_SUFFIX` (11 entrées + fallback vide)
- [x] **1.23** `_normalize_ticker(ticker: str) -> str` — supprime le suffixe d'exchange
- [x] **1.24** `EuronextProvider.__init__` : `cache_dir`, `cache_ttl_seconds`, `timeout`
- [x] **1.25** `resolve_ticker` : appel HTTP + parse JSON + construction `EuronextRecord`
- [x] **1.26** `resolve_isin` : appel HTTP search + parse JSON + construction `EuronextRecord`
- [x] **1.27** Logique de cache (lecture, écriture, TTL, stale fallback) — partagée entre les deux méthodes
- [x] **1.28** `invalidate_cache`
- [x] **1.29** Tous les appels HTTP via `run_in_executor`
- [x] **1.30** `pytest tests/test_euronext_provider.py` → 🟢 tous verts

---

## Étape 2 — MarketDataFacade (SF-2)

### Tests (`tests/test_market_data_facade.py`)

- [x] **2.1** Fixtures : mocks `YahooProvider`, `BoursoramaProvider`, `EuronextProvider`
- [x] **2.2** `get_quote(ticker)` : appels Yahoo et Boursorama lancés en parallèle
- [x] **2.3** `get_quote(isin)` : résolution via `EuronextProvider.resolve_ticker` avant appel Yahoo
- [x] **2.4** Champs dividende/rendement/last_dividend_date : issus de Boursorama quand disponibles
- [x] **2.5** Champs dividende/rendement : fallback Yahoo si Boursorama lève `APIError`
- [x] **2.6** Champs dividende/rendement : fallback Yahoo si Boursorama retourne `None`
- [x] **2.7** `consensus` : issu de Boursorama, absent du résultat si Boursorama échoue
- [x] **2.8** `performance` : issu de Boursorama, liste vide si Boursorama échoue
- [x] **2.9** Champs techniques (cours, historique) : toujours issus de Yahoo
- [x] **2.10** `get_quote` : identifiant non résolvable → lève `APIError`
- [x] **2.11** `get_history` : délègue à Yahoo avec le ticker résolu
- [x] **2.12** `get_news` : délègue à Yahoo avec le ticker résolu
- [x] **2.13** `get_option_chain` : délègue à Yahoo avec le ticker résolu
- [x] **2.14** `get_option_expirations` : délègue à Yahoo avec le ticker résolu
- [x] **2.15** `get_earnings_dates` : délègue à Yahoo avec le ticker résolu
- [x] **2.16** Résolution ISIN : `EuronextProvider.resolve_ticker` appelé une seule fois par identifiant (pas de double appel)
- [x] **2.17** Boursorama reçoit le ticker court (sans suffixe `.PA`) ou l'ISIN selon ce qui est disponible
- [x] **2.18** `get_quote` retourne un dict plat avec tous les champs fusionnés
- [x] **2.19** Exception Yahoo non gérée → propagée (pas de silence silencieux)
- [x] **2.20** `MarketDataFacade` n'importe pas `yfinance` directement

### Code de production (`stockscreen/providers/facade.py`)

- [x] **2.21** `MarketDataFacade.__init__` : injection `yahoo`, `boursorama`, `euronext`
- [x] **2.22** `_resolve_to_ticker(identifier)` : retourne `(yahoo_ticker, isin_or_short)` pour Boursorama
- [x] **2.23** `get_quote` : `asyncio.gather` Yahoo + Boursorama, merge avec priorité Boursorama pour dividende
- [x] **2.24** Méthodes déléguées : `get_history`, `get_news`, `get_option_chain`, `get_option_expirations`, `get_earnings_dates`
- [x] **2.25** `pytest tests/test_market_data_facade.py` → 🟢 tous verts

---

## Étape 3 — Adapter ScreenerService & NewsService (SF-2 suite)

### Tests (mise à jour)

- [x] **3.1** `tests/test_screener_service.py` : remplacer les mocks `YahooProvider` par `MarketDataFacade`
- [x] **3.2** `tests/test_news_service.py` : idem
- [x] **3.3** Vérifier que tous les tests existants passent toujours sans modification de comportement

### Code de production

- [x] **3.4** `services/screener.py` : `provider: YahooProvider` → `provider: MarketDataFacade` (type hint uniquement — duck typing)
- [x] **3.5** `services/news.py` : idem
- [x] **3.6** `server.py` : `create_services()` instancie `EuronextProvider`, `BoursoramaProvider`, `MarketDataFacade` et les injecte
- [x] **3.7** `config.py` : ajouter `EURONEXT_CACHE_TTL_SECONDS`
- [x] **3.8** `pytest` (suite complète) → 🟢 tous verts

---

## Étape 4 — BoursoramaPalmaresScaper (SF-3)

### Exploration préalable (hors TDD)

- [x] **4.0** Vérifier manuellement le paramètre de pagination (`?page=N` ?) et la structure HTML du tableau sur `https://www.boursorama.com/bourse/actions/palmares/dividendes/`

### Tests (`tests/test_boursorama_palmares.py`)

- [x] **4.1** Fixture HTML : page 1 du tableau palmarès (une ligne complète + une ligne avec champs manquants)
- [x] **4.2** Fixture HTML : page avec lien pagination (permet de détecter le nombre total de pages)
- [x] **4.3** Fixture HTML : dernière page (pas de bouton "suivant")
- [x] **4.4** `fetch_page(1)` : retourne une liste de `PalmaresEntry` avec tous les champs renseignés
- [x] **4.5** `fetch_page` : champs manquants → valeurs `None` (pas d'exception)
- [x] **4.6** `fetch_page` : `rendement` parsé en float (ex: `"5,08 %"` → `5.08`)
- [x] **4.7** `fetch_page` : `cours` parsé en float (ex: `"59,42"` → `59.42`)
- [x] **4.8** `fetch_page` : `date_detachement` et `date_paiement` parsées en ISO (`"18/03/2026"` → `"2026-03-18"`)
- [x] **4.9** `fetch_page` : `code_bourso` extrait du href `/cours/{code}/`
- [x] **4.10** `fetch_page` : `secteur` et `compartiment` extraits correctement
- [x] **4.11** `fetch_all` : agrège les résultats de N pages
- [x] **4.12** `fetch_all` : détecte automatiquement le nombre de pages via la pagination HTML
- [x] **4.13** `fetch_all` : erreur sur une page → logger + continuer (best-effort)
- [x] **4.14** Tous les appels HTTP via `run_in_executor` (non bloquant)
- [x] **4.15** La session `requests` utilise des headers navigateur (anti-bot)

### Code de production (`stockscreen/providers/boursorama_palmares.py`)

- [x] **4.16** Dataclass `PalmaresEntry` (ou import depuis `models/schemas.py`)
- [x] **4.17** `BoursoramaPalmaresScaper.__init__` : `session`, `timeout`, `base_url`
- [x] **4.18** `_parse_page(html) -> list[PalmaresEntry]`
- [x] **4.19** `_detect_page_count(html) -> int`
- [x] **4.20** `fetch_page(page: int) -> list[PalmaresEntry]`
- [x] **4.21** `fetch_all() -> list[PalmaresEntry]`
- [x] **4.22** `pytest tests/test_boursorama_palmares.py` → 🟢 tous verts

---

## Étape 5 — PalmaresStore (SF-5)

### Tests (`tests/test_palmares_store.py`)

- [x] **5.1** `save` + `load` : round-trip sans perte de données
- [x] **5.2** `load` : retourne `None` si le fichier n'existe pas
- [x] **5.3** `save` : crée le répertoire `data/palmares/` si absent
- [x] **5.4** `load` : désérialise correctement les types (`float`, `str | None`, liste)
- [x] **5.5** `save` : le fichier JSON est lisible par un humain (indentation)
- [x] **5.6** `save` écrase le snapshot précédent (pas d'accumulation)
- [x] **5.7** `load` : fichier JSON corrompu → retourne `None` (pas d'exception)

### Code de production (`stockscreen/store/palmares_store.py`)

- [x] **5.8** `PalmaresStore.__init__` : `base_path`
- [x] **5.9** `_path() -> str` : `{base_path}/palmares/palmares_dividendes.json`
- [x] **5.10** `save(snapshot: PalmaresSnapshot) -> None`
- [x] **5.11** `load() -> PalmaresSnapshot | None`
- [x] **5.12** `pytest tests/test_palmares_store.py` → 🟢 tous verts

---

## Étape 6 — PalmaresService (SF-4)

### Tests (`tests/test_palmares_service.py`)

- [x] **6.1** `get()` : cache frais → retourne le snapshot sans appel au scraper
- [x] **6.2** `get()` : cache expiré → déclenche `scraper.fetch_all()`
- [x] **6.3** `get()` : cache absent → déclenche `scraper.fetch_all()`
- [x] **6.4** `refresh()` : force `scraper.fetch_all()` même si cache frais
- [x] **6.5** `get(min_rendement=3.0)` : filtre les entrées avec `rendement < 3.0`
- [x] **6.6** `get(max_rendement=5.0)` : filtre les entrées avec `rendement > 5.0`
- [x] **6.7** `get(secteur="Energie")` : filtre exact sur le secteur (insensible à la casse)
- [x] **6.8** `get(compartiment="A")` : filtre exact sur le compartiment
- [x] **6.9** `get(nom_contains="total")` : filtre partiel insensible à la casse
- [x] **6.10** `get(limit=10)` : retourne au plus 10 entrées
- [x] **6.11** Résultat trié par `rendement` décroissant
- [x] **6.12** Entrée avec `rendement = None` : placée en fin de liste lors du tri
- [x] **6.13** `get()` : snapshot sauvegardé via `PalmaresStore.save()` après scraping
- [x] **6.14** `refresh()` : retourne un `PalmaresSnapshot` avec `total_entries` correct
- [x] **6.15** Filtres combinés : `min_rendement + secteur` s'appliquent ensemble (AND)

### Code de production (`stockscreen/services/palmares_service.py`)

- [x] **6.16** `PalmaresService.__init__` : `scraper`, `store`, `cache_ttl_seconds`
- [x] **6.17** `_is_fresh(snapshot) -> bool` : compare `fetched_at` + TTL
- [x] **6.18** `get(...)` avec tous les filtres et tri
- [x] **6.19** `refresh() -> PalmaresSnapshot`
- [x] **6.20** `pytest tests/test_palmares_service.py` → 🟢 tous verts

---

## Étape 7 — Wiring server + tool `get_palmares` (SF-6)

### Tests (mise à jour `tests/test_server.py`)

- [x] **7.1** `create_services()` retourne un 5-tuple
- [x] **7.2** Tool `get_palmares` : appelle `PalmaresService.get()` avec les bons paramètres
- [x] **7.3** Tool `get_palmares` : `force_refresh=True` → appelle `PalmaresService.refresh()`
- [x] **7.4** Tool `get_palmares` : erreur interne → retourne `{"error": "..."}`
- [x] **7.5** Tool `get_palmares` : `limit` respecté dans la réponse

### Code de production

- [x] **7.6** `models/schemas.py` : ajouter `PalmaresEntry`, `PalmaresSnapshot`
- [x] **7.7** `config.py` : ajouter `PALMARES_CACHE_TTL_SECONDS`
- [x] **7.8** `server.py` : `create_services()` → 5-tuple avec `PalmaresService`
- [x] **7.9** `server.py` : tool `get_palmares` avec tous les paramètres SF-6
- [x] **7.10** `pytest` (suite complète) → 🟢 tous verts

---

## Récapitulatif

| Étape | Fichiers créés | Fichiers modifiés | Tests |
|---|---|---|---|
| 1 — EuronextProvider | `providers/euronext.py` | — | `test_euronext_provider.py` (~20) |
| 2 — MarketDataFacade | `providers/facade.py` | — | `test_market_data_facade.py` (~20) |
| 3 — Adapter services | — | `services/screener.py`, `services/news.py`, `server.py`, `config.py` | Mise à jour existants |
| 4 — PalmaresScaper | `providers/boursorama_palmares.py` | — | `test_boursorama_palmares.py` (~15) |
| 5 — PalmaresStore | `store/palmares_store.py` | — | `test_palmares_store.py` (~7) |
| 6 — PalmaresService | `services/palmares_service.py` | — | `test_palmares_service.py` (~15) |
| 7 — Wiring + tool | — | `server.py`, `config.py`, `models/schemas.py` | Mise à jour `test_server.py` |

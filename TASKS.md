# Tâches d'implémentation — Façade & Palmarès Dividendes

> Référence : [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
> Méthodologie : TDD — les tests sont écrits **avant** le code de production.
> Convention : cocher `[x]` dès qu'une tâche est terminée et les tests passent au vert.

---

## Étape 1 — EuronextProvider (SF-1)

### Tests (`tests/test_euronext_provider.py`)

- [x] **1.1** Fixture : réponse JSON Euronext ISIN→ticker (`XPAR`, `XETR`, `XLON`, `XAMS`, MIC inconnu)
- [ ] **1.2** Fixture : réponse JSON Euronext ticker→ISIN (search)
- [ ] **1.3** `resolve_ticker` : retourne un `EuronextRecord` avec `yahoo_ticker = "TTE.PA"` pour ISIN XPAR
- [ ] **1.4** `resolve_ticker` : suffixe `.DE` pour MIC `XETR`
- [ ] **1.5** `resolve_ticker` : suffixe vide pour MIC inconnu
- [ ] **1.6** `resolve_ticker` : retourne `None` si ISIN non trouvé (réponse vide)
- [ ] **1.7** `resolve_isin` : retourne un `EuronextRecord` depuis ticker `"TTE"`
- [ ] **1.8** `resolve_isin` : normalise le ticker — `"TTE.PA"` → appel avec `"TTE"`
- [ ] **1.9** `resolve_isin` : retourne `None` si ticker non trouvé
- [ ] **1.10** Cache : `resolve_ticker` sur ISIN déjà en cache ne fait pas d'appel HTTP
- [ ] **1.11** Cache : `resolve_isin` sur ticker déjà en cache ne fait pas d'appel HTTP
- [ ] **1.12** Cache : cache expiré (TTL dépassé) déclenche un nouvel appel HTTP
- [ ] **1.13** Cache : fichier partagé entre `resolve_ticker` et `resolve_isin` pour le même record
- [ ] **1.14** Cache : erreur réseau → stale fallback si cache disponible
- [ ] **1.15** Cache : erreur réseau sans cache → retourne `None` (pas d'exception)
- [ ] **1.16** `invalidate_cache` : supprime le fichier cache (ISIN ou ticker)
- [ ] **1.17** `invalidate_cache` : ne lève pas d'exception si le fichier n'existe pas
- [ ] **1.18** Tous les appels HTTP s'exécutent dans `run_in_executor` (non bloquant)
- [ ] **1.19** `EuronextRecord` : `yahoo_ticker` construit correctement pour chaque MIC de la table
- [ ] **1.20** `EuronextRecord` : `cached_at` est un ISO timestamp valide

### Code de production (`stockscreen/providers/euronext.py`)

- [ ] **1.21** Dataclass `EuronextRecord` avec tous les champs
- [ ] **1.22** Table `_MIC_TO_SUFFIX` (11 entrées + fallback vide)
- [ ] **1.23** `_normalize_ticker(ticker: str) -> str` — supprime le suffixe d'exchange
- [ ] **1.24** `EuronextProvider.__init__` : `cache_dir`, `cache_ttl_seconds`, `timeout`
- [ ] **1.25** `resolve_ticker` : appel HTTP + parse JSON + construction `EuronextRecord`
- [ ] **1.26** `resolve_isin` : appel HTTP search + parse JSON + construction `EuronextRecord`
- [ ] **1.27** Logique de cache (lecture, écriture, TTL, stale fallback) — partagée entre les deux méthodes
- [ ] **1.28** `invalidate_cache`
- [ ] **1.29** Tous les appels HTTP via `run_in_executor`
- [ ] **1.30** `pytest tests/test_euronext_provider.py` → 🟢 tous verts

---

## Étape 2 — MarketDataFacade (SF-2)

### Tests (`tests/test_market_data_facade.py`)

- [ ] **2.1** Fixtures : mocks `YahooProvider`, `BoursoramaProvider`, `EuronextProvider`
- [ ] **2.2** `get_quote(ticker)` : appels Yahoo et Boursorama lancés en parallèle
- [ ] **2.3** `get_quote(isin)` : résolution via `EuronextProvider.resolve_ticker` avant appel Yahoo
- [ ] **2.4** Champs dividende/rendement/last_dividend_date : issus de Boursorama quand disponibles
- [ ] **2.5** Champs dividende/rendement : fallback Yahoo si Boursorama lève `APIError`
- [ ] **2.6** Champs dividende/rendement : fallback Yahoo si Boursorama retourne `None`
- [ ] **2.7** `consensus` : issu de Boursorama, absent du résultat si Boursorama échoue
- [ ] **2.8** `performance` : issu de Boursorama, liste vide si Boursorama échoue
- [ ] **2.9** Champs techniques (cours, historique) : toujours issus de Yahoo
- [ ] **2.10** `get_quote` : identifiant non résolvable → lève `APIError`
- [ ] **2.11** `get_history` : délègue à Yahoo avec le ticker résolu
- [ ] **2.12** `get_news` : délègue à Yahoo avec le ticker résolu
- [ ] **2.13** `get_option_chain` : délègue à Yahoo avec le ticker résolu
- [ ] **2.14** `get_option_expirations` : délègue à Yahoo avec le ticker résolu
- [ ] **2.15** `get_earnings_dates` : délègue à Yahoo avec le ticker résolu
- [ ] **2.16** Résolution ISIN : `EuronextProvider.resolve_ticker` appelé une seule fois par identifiant (pas de double appel)
- [ ] **2.17** Boursorama reçoit le ticker court (sans suffixe `.PA`) ou l'ISIN selon ce qui est disponible
- [ ] **2.18** `get_quote` retourne un dict plat avec tous les champs fusionnés
- [ ] **2.19** Exception Yahoo non gérée → propagée (pas de silence silencieux)
- [ ] **2.20** `MarketDataFacade` n'importe pas `yfinance` directement

### Code de production (`stockscreen/providers/facade.py`)

- [ ] **2.21** `MarketDataFacade.__init__` : injection `yahoo`, `boursorama`, `euronext`
- [ ] **2.22** `_resolve_to_ticker(identifier)` : retourne `(yahoo_ticker, isin_or_short)` pour Boursorama
- [ ] **2.23** `get_quote` : `asyncio.gather` Yahoo + Boursorama, merge avec priorité Boursorama pour dividende
- [ ] **2.24** Méthodes déléguées : `get_history`, `get_news`, `get_option_chain`, `get_option_expirations`, `get_earnings_dates`
- [ ] **2.25** `pytest tests/test_market_data_facade.py` → 🟢 tous verts

---

## Étape 3 — Adapter ScreenerService & NewsService (SF-2 suite)

### Tests (mise à jour)

- [ ] **3.1** `tests/test_screener_service.py` : remplacer les mocks `YahooProvider` par `MarketDataFacade`
- [ ] **3.2** `tests/test_news_service.py` : idem
- [ ] **3.3** Vérifier que tous les tests existants passent toujours sans modification de comportement

### Code de production

- [ ] **3.4** `services/screener.py` : `provider: YahooProvider` → `provider: MarketDataFacade` (type hint uniquement — duck typing)
- [ ] **3.5** `services/news.py` : idem
- [ ] **3.6** `server.py` : `create_services()` instancie `EuronextProvider`, `BoursoramaProvider`, `MarketDataFacade` et les injecte
- [ ] **3.7** `config.py` : ajouter `EURONEXT_CACHE_TTL_SECONDS`
- [ ] **3.8** `pytest` (suite complète) → 🟢 tous verts

---

## Étape 4 — BoursoramaPalmaresScaper (SF-3)

### Exploration préalable (hors TDD)

- [ ] **4.0** Vérifier manuellement le paramètre de pagination (`?page=N` ?) et la structure HTML du tableau sur `https://www.boursorama.com/bourse/actions/palmares/dividendes/`

### Tests (`tests/test_boursorama_palmares.py`)

- [ ] **4.1** Fixture HTML : page 1 du tableau palmarès (une ligne complète + une ligne avec champs manquants)
- [ ] **4.2** Fixture HTML : page avec lien pagination (permet de détecter le nombre total de pages)
- [ ] **4.3** Fixture HTML : dernière page (pas de bouton "suivant")
- [ ] **4.4** `fetch_page(1)` : retourne une liste de `PalmaresEntry` avec tous les champs renseignés
- [ ] **4.5** `fetch_page` : champs manquants → valeurs `None` (pas d'exception)
- [ ] **4.6** `fetch_page` : `rendement` parsé en float (ex: `"5,08 %"` → `5.08`)
- [ ] **4.7** `fetch_page` : `cours` parsé en float (ex: `"59,42"` → `59.42`)
- [ ] **4.8** `fetch_page` : `date_detachement` et `date_paiement` parsées en ISO (`"18/03/2026"` → `"2026-03-18"`)
- [ ] **4.9** `fetch_page` : `code_bourso` extrait du href `/cours/{code}/`
- [ ] **4.10** `fetch_page` : `secteur` et `compartiment` extraits correctement
- [ ] **4.11** `fetch_all` : agrège les résultats de N pages
- [ ] **4.12** `fetch_all` : détecte automatiquement le nombre de pages via la pagination HTML
- [ ] **4.13** `fetch_all` : erreur sur une page → logger + continuer (best-effort)
- [ ] **4.14** Tous les appels HTTP via `run_in_executor` (non bloquant)
- [ ] **4.15** La session `requests` utilise des headers navigateur (anti-bot)

### Code de production (`stockscreen/providers/boursorama_palmares.py`)

- [ ] **4.16** Dataclass `PalmaresEntry` (ou import depuis `models/schemas.py`)
- [ ] **4.17** `BoursoramaPalmaresScaper.__init__` : `session`, `timeout`, `base_url`
- [ ] **4.18** `_parse_page(html) -> list[PalmaresEntry]`
- [ ] **4.19** `_detect_page_count(html) -> int`
- [ ] **4.20** `fetch_page(page: int) -> list[PalmaresEntry]`
- [ ] **4.21** `fetch_all() -> list[PalmaresEntry]`
- [ ] **4.22** `pytest tests/test_boursorama_palmares.py` → 🟢 tous verts

---

## Étape 5 — PalmaresStore (SF-5)

### Tests (`tests/test_palmares_store.py`)

- [ ] **5.1** `save` + `load` : round-trip sans perte de données
- [ ] **5.2** `load` : retourne `None` si le fichier n'existe pas
- [ ] **5.3** `save` : crée le répertoire `data/palmares/` si absent
- [ ] **5.4** `load` : désérialise correctement les types (`float`, `str | None`, liste)
- [ ] **5.5** `save` : le fichier JSON est lisible par un humain (indentation)
- [ ] **5.6** `save` écrase le snapshot précédent (pas d'accumulation)
- [ ] **5.7** `load` : fichier JSON corrompu → retourne `None` (pas d'exception)

### Code de production (`stockscreen/store/palmares_store.py`)

- [ ] **5.8** `PalmaresStore.__init__` : `base_path`
- [ ] **5.9** `_path() -> str` : `{base_path}/palmares/palmares_dividendes.json`
- [ ] **5.10** `save(snapshot: PalmaresSnapshot) -> None`
- [ ] **5.11** `load() -> PalmaresSnapshot | None`
- [ ] **5.12** `pytest tests/test_palmares_store.py` → 🟢 tous verts

---

## Étape 6 — PalmaresService (SF-4)

### Tests (`tests/test_palmares_service.py`)

- [ ] **6.1** `get()` : cache frais → retourne le snapshot sans appel au scraper
- [ ] **6.2** `get()` : cache expiré → déclenche `scraper.fetch_all()`
- [ ] **6.3** `get()` : cache absent → déclenche `scraper.fetch_all()`
- [ ] **6.4** `refresh()` : force `scraper.fetch_all()` même si cache frais
- [ ] **6.5** `get(min_rendement=3.0)` : filtre les entrées avec `rendement < 3.0`
- [ ] **6.6** `get(max_rendement=5.0)` : filtre les entrées avec `rendement > 5.0`
- [ ] **6.7** `get(secteur="Energie")` : filtre exact sur le secteur (insensible à la casse)
- [ ] **6.8** `get(compartiment="A")` : filtre exact sur le compartiment
- [ ] **6.9** `get(nom_contains="total")` : filtre partiel insensible à la casse
- [ ] **6.10** `get(limit=10)` : retourne au plus 10 entrées
- [ ] **6.11** Résultat trié par `rendement` décroissant
- [ ] **6.12** Entrée avec `rendement = None` : placée en fin de liste lors du tri
- [ ] **6.13** `get()` : snapshot sauvegardé via `PalmaresStore.save()` après scraping
- [ ] **6.14** `refresh()` : retourne un `PalmaresSnapshot` avec `total_entries` correct
- [ ] **6.15** Filtres combinés : `min_rendement + secteur` s'appliquent ensemble (AND)

### Code de production (`stockscreen/services/palmares_service.py`)

- [ ] **6.16** `PalmaresService.__init__` : `scraper`, `store`, `cache_ttl_seconds`
- [ ] **6.17** `_is_fresh(snapshot) -> bool` : compare `fetched_at` + TTL
- [ ] **6.18** `get(...)` avec tous les filtres et tri
- [ ] **6.19** `refresh() -> PalmaresSnapshot`
- [ ] **6.20** `pytest tests/test_palmares_service.py` → 🟢 tous verts

---

## Étape 7 — Wiring server + tool `get_palmares` (SF-6)

### Tests (mise à jour `tests/test_server.py`)

- [ ] **7.1** `create_services()` retourne un 5-tuple
- [ ] **7.2** Tool `get_palmares` : appelle `PalmaresService.get()` avec les bons paramètres
- [ ] **7.3** Tool `get_palmares` : `force_refresh=True` → appelle `PalmaresService.refresh()`
- [ ] **7.4** Tool `get_palmares` : erreur interne → retourne `{"error": "..."}`
- [ ] **7.5** Tool `get_palmares` : `limit` respecté dans la réponse

### Code de production

- [ ] **7.6** `models/schemas.py` : ajouter `PalmaresEntry`, `PalmaresSnapshot`
- [ ] **7.7** `config.py` : ajouter `PALMARES_CACHE_TTL_SECONDS`
- [ ] **7.8** `server.py` : `create_services()` → 5-tuple avec `PalmaresService`
- [ ] **7.9** `server.py` : tool `get_palmares` avec tous les paramètres SF-6
- [ ] **7.10** `pytest` (suite complète) → 🟢 tous verts

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

# Plan d'implémentation — Façade & Palmarès Dividendes

> **Version** : 1.0 — 2026-03-24
> **Scope** : `EuronextProvider`, `MarketDataFacade`, `BoursoramaPalmaresScaper`, `PalmaresService`, tool `get_palmares`

---

## Spécifications fonctionnelles

### SF-1 — Résolution ISIN ↔ Ticker (EuronextProvider)

**Besoin** : Résolution bidirectionnelle entre ISIN et ticker Yahoo Finance.
- `resolve_ticker(isin)` → depuis `FR0000131104`, obtenir `TTE.PA`
- `resolve_isin(ticker)` → depuis `TTE` ou `TTE.PA`, obtenir `FR0000131104`

**Règles métier** :

**Direction ISIN → ticker** (`resolve_ticker`) :
- Source : API JSON Euronext Live
  `https://live.euronext.com/api/quotes/{isin}/intraday_ioapi/2?fieldlist=isin,symbol,name,mic`
- Le suffixe Yahoo est dérivé du champ `mic` selon la table de correspondance :

  | MIC | Suffixe |
  |---|---|
  | XPAR | .PA |
  | XETR | .DE |
  | XLON | .L |
  | XAMS | .AS |
  | XMIL | .MI |
  | XMAD | .MC |
  | XBRU | .BR |
  | XLIS | .LS |
  | XHEL | .HE |
  | XSTO | .ST |
  | XOSL | .OL |
  | *(autre)* | *(vide — ticker brut sans suffixe)* |

**Direction ticker → ISIN** (`resolve_isin`) :
- Le ticker est normalisé avant l'appel : suppression du suffixe d'exchange (`.PA`, `.DE`, etc.)
- Source : API de recherche Euronext
  `https://live.euronext.com/search_instruments/{symbol}` — retourne un tableau JSON dont le premier élément correspondant contient le champ `isin`
- ⚠️ À confirmer à l'implémentation (endpoint et format de réponse exact).
- En cas d'ambiguïté (même ticker sur plusieurs places), retourner le premier résultat Euronext.

**Règles communes** :
- Cache JSON par clé (ISIN ou ticker normalisé), TTL configurable (défaut 7 jours).
- Un seul fichier cache par `EuronextRecord` — partagé entre les deux directions.
- En cas d'échec réseau, retourner la valeur en cache si disponible (stale fallback).
- Retourne `None` si la clé est inconnue d'Euronext.

**Interface publique** :
```python
class EuronextProvider:
    async def resolve_ticker(self, isin: str) -> EuronextRecord | None
    # ISIN → EuronextRecord (contient yahoo_ticker)

    async def resolve_isin(self, ticker: str) -> EuronextRecord | None
    # ticker ("TTE" ou "TTE.PA") → EuronextRecord (contient isin)

    def invalidate_cache(self, key: str) -> None
    # key = ISIN ou ticker normalisé

@dataclass
class EuronextRecord:
    isin: str
    symbol: str       # ex: "TTE"  (sans suffixe)
    name: str         # ex: "TotalEnergies SE"
    mic: str          # ex: "XPAR"
    yahoo_ticker: str # ex: "TTE.PA"
    cached_at: str    # ISO timestamp
```

---

### SF-2 — Façade de données de marché (MarketDataFacade)

**Besoin** : Point d'entrée unique pour les services. Cache les appels croisés, orchestre les providers, applique les stratégies de fallback.

**Règles métier** :

| Champ | Source primaire | Fallback |
|---|---|---|
| `dividende`, `rendement`, `last_dividend_date` | `BoursoramaProvider` | `YahooProvider` |
| `consensus` | `BoursoramaProvider` | — (None) |
| `performance` (CA/RN) | `BoursoramaProvider` | — (liste vide) |
| `cours`, `historique`, `options`, `news` | `YahooProvider` | — |
| `ticker_yahoo` | `EuronextProvider` | — |

- L'entrée peut être un **ticker Yahoo** (`TTE.PA`) ou un **ISIN** (`FR0000131104`).
- Si l'entrée est un ticker et que Boursorama est demandé, la résolution se fait via `EuronextProvider` (ticker → MIC → ISIN possible uniquement si déjà en cache ; sinon Boursorama reçoit le ticker court sans suffixe).
- Si l'entrée est un ISIN, `EuronextProvider` fournit le ticker Yahoo.
- Boursorama n'est sollicité que pour les champs qui lui appartiennent (pas d'appel inutile).
- Les appels Boursorama et Yahoo sont lancés en parallèle via `asyncio.gather` quand les deux sont nécessaires.
- Le résultat est un dict plat `MarketData` (pas un dataclass — compatible avec le screener existant).

**Interface publique** :
```python
class MarketDataFacade:
    def __init__(
        self,
        yahoo: YahooProvider,
        boursorama: BoursoramaProvider,
        euronext: EuronextProvider,
    ): ...

    async def get_quote(self, identifier: str) -> MarketData
    # identifier = ticker Yahoo ("TTE.PA") ou ISIN ("FR0000131104")

    async def get_history(self, identifier: str, period: str = "1y") -> pd.DataFrame
    async def get_news(self, identifier: str) -> list[dict]
    async def get_option_chain(self, identifier: str, expiry: str) -> Any
    async def get_option_expirations(self, identifier: str) -> tuple
    async def get_earnings_dates(self, identifier: str) -> dict
```

`MarketData` est un `TypedDict` (ou dict) contenant tous les champs issus des providers.

---

### SF-3 — Scraping du palmarès dividendes Boursorama (BoursoramaPalmaresScaper)

**Source** : `https://www.boursorama.com/bourse/actions/palmares/dividendes/`

**Besoin** : Scraper toutes les pages du palmarès et retourner une liste structurée.

**Règles métier** :
- Pagination : détecter le nombre total de pages via le composant de pagination HTML, scraper séquentiellement (pour ne pas surcharger le serveur).
- URL de page N : `?page=N` (à confirmer à l'implémentation).
- Par ligne du tableau, extraire :

  | Champ | Source HTML | Type |
  |---|---|---|
  | `isin` | Lien de la fiche valeur (extrait du href `/cours/{code}/` via lookup Boursorama) | str |
  | `code_bourso` | href `/cours/{code}/` | str |
  | `nom` | Colonne nom | str |
  | `secteur` | Colonne secteur | str \| None |
  | `compartiment` | Colonne compartiment (A / B / C) | str \| None |
  | `cours` | Colonne cours | float \| None |
  | `dividende` | Colonne dividende | float \| None |
  | `rendement` | Colonne rendement (%) | float \| None |
  | `date_detachement` | Colonne date détachement | str \| None (ISO) |
  | `date_paiement` | Colonne date paiement | str \| None (ISO) |

- L'ISIN n'est pas dans le tableau — il sera enrichi en option via `EuronextProvider` (hors scope du scraper lui-même).
- En cas d'erreur sur une page, logger et continuer (best-effort).
- Retourne `list[PalmaresEntry]`.

**Interface publique** :
```python
@dataclass
class PalmaresEntry:
    code_bourso: str
    nom: str
    secteur: str | None
    compartiment: str | None
    cours: float | None
    dividende: float | None
    rendement: float | None
    date_detachement: str | None
    date_paiement: str | None
    isin: str | None = None     # enrichi ultérieurement

class BoursoramaPalmaresScaper:
    def __init__(self, session: requests.Session | None = None, timeout: int = 10): ...
    async def fetch_all(self) -> list[PalmaresEntry]
    async def fetch_page(self, page: int) -> list[PalmaresEntry]
```

---

### SF-4 — Service palmarès (PalmaresService)

**Besoin** : Orchestrer le scraping, stocker le snapshot, exposer des filtres.

**Règles métier** :
- Un snapshot = liste complète `list[PalmaresEntry]` + métadonnées (`fetched_at`, `page_count`, `total_entries`).
- Le snapshot est persisté dans `data/palmares/palmares_dividendes.json`.
- TTL configurable (défaut 24h). Si le cache est frais, retourner directement sans rescraper.
- Filtres applicables : `min_rendement`, `max_rendement`, `secteur`, `compartiment`, `nom_contains`.
- Tri : par `rendement` décroissant par défaut.
- Méthode `refresh()` force un nouveau scraping indépendamment du TTL.

**Interface publique** :
```python
class PalmaresService:
    def __init__(
        self,
        scraper: BoursoramaPalmaresScaper,
        store: PalmaresStore,
        cache_ttl_seconds: float = 86400,
    ): ...

    async def get(
        self,
        min_rendement: float | None = None,
        max_rendement: float | None = None,
        secteur: str | None = None,
        compartiment: str | None = None,
        nom_contains: str | None = None,
        limit: int = 100,
    ) -> PalmaresSnapshot

    async def refresh(self) -> PalmaresSnapshot
```

---

### SF-5 — Persistence (PalmaresStore)

**Besoin** : Lire/écrire le snapshot palmarès sur disque.

**Règles métier** :
- Fichier unique `{data_path}/palmares/palmares_dividendes.json`.
- Format : `{ "fetched_at": "...", "page_count": N, "total_entries": N, "entries": [...] }`.
- Si le fichier n'existe pas, `load()` retourne `None`.

**Interface publique** :
```python
class PalmaresStore:
    def __init__(self, base_path: str): ...
    def load(self) -> PalmaresSnapshot | None
    def save(self, snapshot: PalmaresSnapshot) -> None
```

---

### SF-6 — Tool MCP `get_palmares`

**Besoin** : Exposer le palmarès dividendes comme outil FastMCP.

**Paramètres** :
```
min_rendement (float | None)    — rendement minimum en % (ex: 3.0)
max_rendement (float | None)    — rendement maximum en %
secteur (str | None)            — filtre exact sur le secteur
compartiment (str | None)       — "A", "B" ou "C"
nom_contains (str | None)       — recherche partielle sur le nom (insensible à la casse)
limit (int)                     — max entrées retournées (défaut 50, max 500)
force_refresh (bool)            — forcer un nouveau scraping (défaut false)
```

**Réponse** :
```json
{
  "fetched_at": "2026-03-24T10:00:00",
  "total_entries": 120,
  "returned": 45,
  "entries": [
    {
      "code_bourso": "1rTTE",
      "nom": "TotalEnergies SE",
      "secteur": "Energie",
      "compartiment": "A",
      "cours": 59.42,
      "dividende": 3.02,
      "rendement": 5.08,
      "date_detachement": "2026-03-18",
      "date_paiement": "2026-03-20",
      "isin": null
    }
  ]
}
```

---

## Spécifications techniques

### ST-1 — Arborescence finale

```
stockscreen/
├── providers/
│   ├── yahoo.py                      (existant)
│   ├── boursorama.py                 (existant)
│   ├── euronext.py                   (nouveau — SF-1)
│   ├── facade.py                     (nouveau — SF-2)
│   ├── boursorama_palmares.py        (nouveau — SF-3)
│   └── symbol_fetchers/              (existant)
├── services/
│   ├── screener.py                   (adapter — utilise MarketDataFacade)
│   ├── news.py                       (adapter — utilise MarketDataFacade)
│   ├── watchlist.py                  (inchangé)
│   ├── symbol_service.py             (inchangé)
│   └── palmares_service.py           (nouveau — SF-4)
├── store/
│   ├── data_store.py                 (inchangé)
│   └── palmares_store.py             (nouveau — SF-5)
├── models/
│   └── schemas.py                    (ajouter PalmaresEntry, PalmaresSnapshot)
├── config.py                         (ajouter PALMARES_CACHE_TTL_SECONDS)
└── server.py                         (ajouter create_services, tool get_palmares)
```

### ST-2 — Modèles de données

```python
# models/schemas.py (ajouts)

@dataclass
class PalmaresEntry:
    code_bourso: str
    nom: str
    secteur: str | None
    compartiment: str | None
    cours: float | None
    dividende: float | None
    rendement: float | None
    date_detachement: str | None
    date_paiement: str | None
    isin: str | None = None

@dataclass
class PalmaresSnapshot:
    fetched_at: str           # ISO datetime
    page_count: int
    total_entries: int
    entries: list[PalmaresEntry]
```

### ST-3 — Stratégie de test (TDD)

Chaque composant est développé **tests d'abord**. Aucun appel réseau réel dans les tests — tout est mocké via `unittest.mock.patch`.

| Fichier test | Composant testé | Nb tests estimés |
|---|---|---|
| `test_euronext_provider.py` | `EuronextProvider` | ~20 |
| `test_market_data_facade.py` | `MarketDataFacade` | ~25 |
| `test_boursorama_palmares.py` | `BoursoramaPalmaresScaper` | ~20 |
| `test_palmares_service.py` | `PalmaresService` | ~15 |
| `test_palmares_store.py` | `PalmaresStore` | ~8 |
| `test_server.py` | tool `get_palmares` + wiring | ~5 (ajout) |

Fixtures HTML : extraits représentatifs des pages Euronext et Boursorama, stockés en constantes dans chaque fichier test.

### ST-4 — Configuration (ajouts dans `config.py`)

```python
PALMARES_CACHE_TTL_SECONDS: float = float(
    os.environ.get("STOCKSCREEN_PALMARES_CACHE_TTL", "86400")
)
EURONEXT_CACHE_TTL_SECONDS: float = float(
    os.environ.get("STOCKSCREEN_EURONEXT_CACHE_TTL", str(7 * 86400))  # 7 jours
)
```

### ST-5 — Wiring `server.py`

`create_services()` passe de 4-tuple à 5-tuple :
```python
def create_services() -> tuple[
    ScreenerService, WatchlistService, NewsService, SymbolService, PalmaresService
]: ...
```

`ScreenerService` et `NewsService` reçoivent `MarketDataFacade` au lieu de `YahooProvider` directement. La façade implémente la même interface que `YahooProvider` pour les méthodes existantes (`get_ticker_info`, `get_history`, `get_news`, `get_option_chain`, `get_option_expirations`, `get_earnings_dates`) — **compatibilité descendante assurée**.

---

## Ordre d'implémentation

```
Étape 1 — EuronextProvider + tests
  └── providers/euronext.py
  └── tests/test_euronext_provider.py

Étape 2 — MarketDataFacade + tests
  └── providers/facade.py
  └── tests/test_market_data_facade.py

Étape 3 — Adapter ScreenerService + NewsService
  └── services/screener.py  (provider: YahooProvider → MarketDataFacade)
  └── services/news.py      (idem)
  └── tests/test_screener_service.py  (mise à jour mocks)
  └── tests/test_news_service.py      (idem)

Étape 4 — BoursoramaPalmaresScaper + tests
  └── providers/boursorama_palmares.py
  └── tests/test_boursorama_palmares.py

Étape 5 — PalmaresStore + tests
  └── store/palmares_store.py
  └── tests/test_palmares_store.py

Étape 6 — PalmaresService + tests
  └── services/palmares_service.py
  └── tests/test_palmares_service.py

Étape 7 — Wiring server + tool get_palmares
  └── stockscreen/server.py
  └── stockscreen/config.py
  └── tests/test_server.py  (mise à jour)
```

---

## Contraintes et points d'attention

- **Boursorama anti-bot** : utiliser la `Session` avec headers navigateur (déjà en place dans `BoursoramaProvider`). Le scraper palmarès réutilise la même session.
- **Pagination palmarès** : tester si le paramètre de page est `?page=N` ou `?p=N` — à vérifier lors de l'implémentation (ST-3 fixtures).
- **ISIN absent du tableau palmarès** : le champ `isin` de `PalmaresEntry` est `None` par défaut. Un enrichissement optionnel via `EuronextProvider` pourrait être ajouté en v2.
- **Rétrocompatibilité** : `ScreenerService` et `NewsService` continueront de fonctionner si instanciés avec un `YahooProvider` brut (la façade implémente la même interface).
- **Pas de base de données** : toute la persistence reste en JSON fichiers, cohérent avec l'existant.

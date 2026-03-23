"""Exception hierarchy for stockscreen."""


class StockscreenError(Exception):
    pass


class ValidationError(StockscreenError):
    pass


class APIError(StockscreenError):
    pass

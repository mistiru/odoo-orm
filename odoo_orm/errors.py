class OdooConnectionError(Exception):
    pass


class OdooORMError(Exception):
    pass


class MissingField(OdooORMError):
    pass

class OdooConnectionError(Exception):
    pass


class OdooConnectionAlreadyExists(OdooConnectionError):
    pass


class OdooConnectionNotConnected(OdooConnectionError):
    pass


class UnsafeOperationNotAllowed(OdooConnectionError):
    pass


class OdooORMError(Exception):
    pass


class MissingField(OdooORMError):
    pass


class IncompleteModel(OdooORMError):
    pass


class FieldDoesNotExist(OdooORMError):
    pass


class InvalidModelState(OdooORMError):
    pass


class LazyReferenceNotResolved(OdooORMError):
    pass

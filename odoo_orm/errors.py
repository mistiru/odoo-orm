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


class AlreadyAssignedField(OdooORMError):
    """Means that the field has already been assigned to an attribute."""

    def __init__(self):
        pass

    def __str__(self):
        return f'This field has already been assigned to an attribute.'


class MissingValue(OdooORMError):
    """Means that the model on Odoo does not contain any value for this field."""

    def __init__(self, instance, odoo_field_names):
        self.instance = instance
        self.odoo_field_names = odoo_field_names

    def __str__(self):
        return f'Instance "{self.instance}" has no value for fields {self.odoo_field_names} on Odoo.'


class MissingValues(OdooORMError):
    """Means that the model on Odoo does not contain any value for these fields."""

    def __init__(self, instance, odoo_field_names):
        self.instance = instance
        self.odoo_field_names = odoo_field_names

    def __str__(self):
        return f'Instance "{self.instance}" has no value for fields {self.odoo_field_names} on Odoo.'


class FieldDoesNotExist(OdooORMError):
    pass


class InvalidModelState(OdooORMError):
    pass


class LazyReferenceNotResolved(OdooORMError):
    pass

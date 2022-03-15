import re
from base64 import b64decode
from copy import copy
from datetime import date, datetime
from decimal import Decimal
from functools import cached_property, reduce
from typing import Any, Callable, Optional, Type, Union
from zoneinfo import ZoneInfo

from odoo_orm.connection import OdooConnection
from odoo_orm.errors import (
    AlreadyAssignedField, FieldDoesNotExist, InvalidModelState, LazyReferenceNotResolved, MissingValue, MissingValues,
)

connection = OdooConnection.get_connection()

Filter = tuple[str, str, Any]

ODOO_OPERATIONS = {
    'ne': '!=',
    'gt': '>',
    'ge': '>=',
    'lt': '<',
    'le': '<=',
    'in': 'in',
    'not_in': 'not in',
}

C2S_PATTERN = re.compile(r'(?<!^)(?=[A-Z])')


def c2s(s: str) -> str:
    return C2S_PATTERN.sub('.', s).lower()


class Field:
    SAVED_VALUE_NAME_PREFIX = '__saved_'

    def __init__(self, *odoo_field_names: str, nullable=False) -> None:
        self.odoo_field_names = set(odoo_field_names)
        self.nullable = nullable

        self.name: str = None

    def __set_name__(self, owner, name: str) -> None:
        if self.name is not None:
            raise AlreadyAssignedField()

        if name.startswith('_'):
            raise ValueError('Cannot assign a Field to a private or magic attribute.')

        self.name = name

        if not self.odoo_field_names:
            self.odoo_field_names = self.default_odoo_field_names

    def __get__(self, instance, owner):
        if instance is None:
            return self

        sentinel = object()
        value = instance.__dict__.get(self.name, sentinel)
        if value is sentinel:
            if self.name != 'id' and instance.id and hasattr(instance, '_completion_callback'):
                value = self.__get__(instance._completion_callback(), owner)
            else:
                raise AttributeError

        return value

    def __set__(self, instance, value) -> None:
        if value is None and not self.nullable:
            raise ValueError(f'{instance}.{self.name} cannot be None.')

        instance.__dict__[self.name] = value

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        raise NotImplementedError

    @cached_property
    def saved_value_name(self) -> str:
        return f'{Field.SAVED_VALUE_NAME_PREFIX}{self.name}'

    def from_odoo(self, instance, **odoo_values: Any) -> None:
        relevant_odoo_values = {k: v for k, v in odoo_values.items() if k in self.odoo_field_names}
        if len(relevant_odoo_values) < len(self.odoo_field_names):
            return

        value = self.construct(**relevant_odoo_values)

        try:
            setattr(instance, self.name, value)
            self.save(instance)
        except ValueError:
            raise MissingValue(instance, self.odoo_field_names)

    def construct(self, **odoo_values: Any):
        raise NotImplementedError

    def save(self, instance) -> None:
        value = getattr(instance, self.name)
        setattr(instance, self.saved_value_name, copy(value))

    def has_changed(self, instance) -> bool:
        value = getattr(instance, self.name)
        sentinel = object()
        saved_value = getattr(instance, self.saved_value_name, sentinel)
        return value is not sentinel and value != saved_value

    def deconstruct(self, value) -> dict[str, Any]:
        raise NotImplementedError


class SimpleField(Field):

    def __init__(self, odoo_field_name: str = None, /, *, nullable=False) -> None:
        if odoo_field_name is None:
            super().__init__(nullable=nullable)
        else:
            super().__init__(odoo_field_name, nullable=nullable)

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.name}

    @cached_property
    def odoo_field_name(self) -> str:
        return list(self.odoo_field_names)[0]

    def construct(self, **odoo_values: Any):
        value = odoo_values[self.odoo_field_name]
        if value is False:
            return None
        else:
            return self.to_python(value)

    def to_python(self, odoo_value: Any):
        return odoo_value

    def deconstruct(self, value) -> dict[str, Any]:
        if value is not None:
            odoo_value = self.to_odoo(value)
        elif self.nullable:
            odoo_value = False
        else:
            raise ValueError('value cannot be None')

        return {self.odoo_field_name: odoo_value}

    def to_odoo(self, value) -> Any:
        return value


class IntegerField(SimpleField):

    def to_python(self, odoo_value: Any) -> int:
        return int(odoo_value)


class StringField(SimpleField):
    pass


class B64Field(SimpleField):

    def to_python(self, odoo_value: str) -> bytes:
        return b64decode(odoo_value)


class BooleanField(SimpleField):

    def construct(self, **odoo_values: bool) -> bool:
        # Do not check if null
        return bool(odoo_values[self.odoo_field_name])


class DecimalField(SimpleField):

    def to_python(self, odoo_value: Any) -> Decimal:
        return Decimal(str(odoo_value)).quantize(Decimal('.01'))

    def to_odoo(self, value: Decimal) -> float:
        return float(value)


class DateField(SimpleField):
    DATE_FORMAT = '%Y-%m-%d'

    def to_python(self, odoo_value: str) -> date:
        return datetime.strptime(odoo_value, self.DATE_FORMAT).date()

    def to_odoo(self, value: date) -> str:
        return date.strftime(value, self.DATE_FORMAT)


class DatetimeField(SimpleField):
    DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def to_python(self, odoo_value: str) -> datetime:
        return datetime.strptime(odoo_value, self.DATETIME_FORMAT).replace(tzinfo=ZoneInfo('UTC'))

    def to_odoo(self, value: datetime) -> str:
        return datetime.strftime(value.astimezone(ZoneInfo('UTC')), self.DATETIME_FORMAT)


class LazyReference:

    def __init__(self) -> None:
        self.field: Optional[RelatedField] = None
        self.model: Optional[Type[ModelBase]] = None

    def attach(self, field: 'RelatedField') -> None:
        self.field = field
        self._do_resolution()

    def resolve(self, model: Type['ModelBase']) -> None:
        self.model = model
        self._do_resolution()

    def _do_resolution(self) -> None:
        if self.field is not None and self.model is not None:
            self.field.related_model = self.model


def resolves(*refs: LazyReference):
    def wrapper(model: Type['ModelBase']):
        for ref in refs:
            ref.resolve(model)
        return model

    return wrapper


Target = Union[str, LazyReference, Type['ModelBase']]


class RelatedField(SimpleField):

    def __init__(self, odoo_field_name: str = None, /, *, model: Target, nullable=False) -> None:
        super().__init__(odoo_field_name, nullable=nullable)

        if isinstance(model, LazyReference):
            model.attach(self)

        self.related_model = model

    def __set_name__(self, owner, name) -> None:
        super().__set_name__(owner, name)

        if self.related_model == 'self':
            self.related_model = owner


class ModelField(RelatedField):

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {f'{self.name}_id'}

    def from_odoo(self, instance, **odoo_values: Any) -> None:
        super().from_odoo(instance, **odoo_values)

        dummy = getattr(instance, self.name, None)
        if dummy is not None:
            dummy._completion_callback = lambda: self.completion_callback(instance)

    def to_python(self, odoo_value: tuple[int, str]) -> 'ModelBase':
        if isinstance(self.related_model, LazyReference):
            raise LazyReferenceNotResolved(f'Lazy reference for field {self.name} has not been resolved')

        return self.related_model(id=odoo_value[0], name=odoo_value[1])

    def completion_callback(self, instance) -> 'ModelBase':
        dummy = getattr(instance, self.name)
        related = self.related_model.objects.get(id=dummy.id)
        setattr(instance, self.name, related)
        self.save(instance)
        return related

    def to_odoo(self, value: 'ModelBase') -> int:
        return value.id


class ModelListField(RelatedField):

    def __init__(self, odoo_field_name: str = None, /, *, model: Target) -> None:
        super().__init__(odoo_field_name, model=model)

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {f'{self.name}_ids'}

    def from_odoo(self, instance, **odoo_values: Any) -> None:
        super().from_odoo(instance, **odoo_values)

        dummy_list = getattr(instance, self.name, [])
        for dummy in dummy_list:
            # TODO _id parameter is the same for all calls ><
            dummy._completion_callback = lambda: self.completion_callback(instance, dummy.id)

    def to_python(self, odoo_value: list[int]) -> list['ModelBase']:
        if isinstance(self.related_model, LazyReference):
            raise LazyReferenceNotResolved(f'Lazy reference for field {self.name} has not been resolved')

        return [self.related_model(id=_id) for _id in odoo_value]

    def completion_callback(self, instance, _id) -> 'ModelBase':
        dummy_list = getattr(instance, self.name)
        related = self.related_model.objects.get(id=_id)
        related_list = [related if dummy.id == _id else dummy for dummy in dummy_list]
        setattr(instance, self.name, related_list)
        self.save(instance)
        return related

    def to_odoo(self, value):
        return [(6, '_', [i.id for i in value])]


class QuerySet:

    def __init__(self, model: Type['ModelBase']) -> None:
        self.model = model
        self.cache: Optional[list[ModelBase]] = None
        self.filters: list[Filter] = []
        self.options: dict[str, Any] = {}
        self.prefetches: set[str] = set()

    def __iter__(self):
        if self.cache is None:
            self.cache = self._execute()
        return iter(self.cache)

    def __getitem__(self, item: int):
        if self.cache is None:
            self.cache = self._execute()
        return self.cache[item]

    def __len__(self) -> int:
        if self.cache is None:
            self.cache = self._execute()
        return len(self.cache)

    def __eq__(self, other) -> bool:
        if type(self) is type(other):
            return (self.filters == other.filters
                    and self.options == other.options
                    and (self.cache is None and other.cache is None and self.prefetches == other.prefetches
                         or self.cache is not None and self.cache == other.cache))
        return False

    def _execute(self) -> list['ModelBase']:
        if 'fields' not in self.options:
            self.options['fields'] = list(self.model.all_fields_odoo_names())

        if len(self.filters) == 1 and self.filters[0][0] == 'id' and self.filters[0][1] in ('=', 'in'):
            if self.filters[0][1] == '=':
                res = connection.execute(self.model.Meta.name, 'read', [self.filters[0][2]], **self.options)
            elif self.filters[0][1] == 'in':
                res = connection.execute(self.model.Meta.name, 'read', self.filters[0][2], **self.options)
            else:
                raise Exception('Wut?')
        else:
            res = connection.execute(self.model.Meta.name, 'search_read', self.filters, **self.options)

        instances = []
        for data in res:
            for field in self.options['fields']:
                if field not in data:
                    raise FieldDoesNotExist(self.model, field)
            instances.append(self.model.from_odoo(**data))

        self._prefetch(*list(self.prefetches))

        return instances

    def _enhance(self, **kwargs) -> 'QuerySet':
        new = QuerySet(self.model)
        new.filters = self.filters.copy()
        new.options = self.options.copy()
        new.prefetches = self.prefetches.copy()

        for kw, val in kwargs.items():
            if kw == 'limit':
                new.options['limit'] = val
            elif kw == 'filter':
                for name, value in val.items():
                    parts = name.split('__')
                    if len(parts) == 1:
                        field = parts[0]
                        operation = '='
                    else:
                        fields, last = parts[:-1], parts[-1]
                        field = '.'.join(fields)
                        if last in ODOO_OPERATIONS:
                            operation = ODOO_OPERATIONS[last]
                        else:
                            field += f'.{last}'
                            operation = '='
                    new.filters.append((field, operation, value))
            elif kw == 'values':
                new.options['fields'] = list(name for field in val for name in self.model.field_odoo_names(field))
            elif kw == 'order_by':
                new.options['order'] = ', '.join(field.lstrip('-') + (' desc' if field.startswith('-') else ' asc')
                                                 for field in val)
            elif kw == 'prefetch':
                new.prefetches |= set(val)
            else:
                raise Exception(f'Argument not recognized {kw}')

        return new

    def filter(self, **kwargs) -> 'QuerySet':
        return self._enhance(filter=kwargs)

    def values(self, *fields: str) -> 'QuerySet':
        return self._enhance(values=fields)

    def limit(self, number: int) -> 'QuerySet':
        return self._enhance(limit=number)

    def order_by(self, *fields: str) -> 'QuerySet':
        return self._enhance(order_by=fields)

    def prefetch(self, *field_names: str) -> 'QuerySet':
        return self._enhance(prefetch=field_names)

    def get(self, **kwargs) -> 'ModelBase':
        if kwargs:
            instances = self.filter(**kwargs)._execute()
        else:
            if self.cache is None:
                self.cache = self._execute()
            instances = self.cache

        if len(instances) == 0:
            raise self.model.DoesNotExist()
        elif len(instances) > 1:
            raise self.model.MultipleObjectsReturned()

        return instances[0]

    def delete(self) -> None:
        self.options['fields'] = ['id']

        res = connection.execute(self.model.Meta.name, 'search_read', self.filters, **self.options)
        ids = [r['id'] for r in res]

        res = connection.execute(self.model.Meta.name, 'unlink', ids)
        if res is not True:
            raise Exception(res)

    def _prefetch(self, *field_names: str) -> None:
        # Extract field names related to this prefetch
        #  merge subsequent ones according to their prefix
        followings_by_field_name: dict[str, list[str]] = {}
        for field_name in sorted(field_names):
            if '__' in field_name:
                field_name, following = field_name.split('__', maxsplit=1)
                followings_by_field_name.setdefault(field_name, []).append(following)
            else:
                followings_by_field_name[field_name] = []

        for field_name in followings_by_field_name:
            followings = followings_by_field_name.get(field_name)
            field = self.model.fields[field_name]

            if isinstance(field, ModelField):
                all_ids = set(getattr(instance, field.name).id for instance in self if getattr(instance, field_name))

                related = field.related_model.objects.filter(id__in=sorted(all_ids))
                if followings:
                    related = related.prefetch(*followings)

                for instance in self:
                    dummy = getattr(instance, field_name)
                    if dummy is None:
                        continue
                    rel = next(r for r in related if r.id == dummy.id)
                    setattr(instance, field_name, rel)
                    field.save(instance)
            elif isinstance(field, ModelListField):
                all_ids = set()
                for instance in self:
                    all_ids |= set([dummy.id for dummy in getattr(instance, field_name)])

                related = field.related_model.objects.filter(id__in=sorted(all_ids))
                if followings:
                    related = related.prefetch(*followings)

                for instance in self:
                    r_ids = [dummy.id for dummy in getattr(instance, field_name)]
                    rels = [r for r in related if r.id in r_ids]
                    setattr(instance, field_name, rels)
                    field.save(instance)
            else:
                raise ValueError('Only support prefetch on RelatedFields')


class Manager:

    def __init__(self, model: Type['ModelBase']) -> None:
        self.model = model

    def __getattr__(self, item: str) -> Callable[[], QuerySet]:
        queryset = self.get_queryset()

        if item == 'all':
            return lambda: queryset
        else:
            return getattr(queryset, item)

    def get_queryset(self) -> QuerySet:
        return QuerySet(self.model)


class MetaModel(type):

    def __init__(cls, name: str, bases: tuple[type], attrs: dict[str, Any]) -> None:
        super().__init__(name, bases, attrs)

        cls.fields = {}
        if name != 'ModelBase':
            for base in reversed(bases):
                if issubclass(base, ModelBase):
                    for attr_name, field in base.fields.items():
                        cls.fields[attr_name] = field
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field):
                cls.fields[attr_name] = attr_value

        if 'objects' not in cls.__dict__:
            cls.objects = Manager(cls)

        if 'Meta' not in cls.__dict__:
            cls.Meta = type('Meta', (), {})
        if 'name' not in cls.Meta.__dict__:
            cls.Meta.name = c2s(name)


class ModelBase(metaclass=MetaModel):
    fields: dict[str, Field]
    objects: Manager

    id = IntegerField()
    name = StringField()

    class Meta:
        name: str

    class DoesNotExist(Exception):
        pass

    class MultipleObjectsReturned(Exception):
        pass

    def __init__(self, **values: Any) -> None:
        for field in self.fields.values():
            if isinstance(field, RelatedField) and isinstance(field.related_model, LazyReference):
                raise LazyReferenceNotResolved(f'Lazy reference for field {field.name} has not been resolved')

        for field_name, value in values.items():
            setattr(self, field_name, value)

    def __eq__(self, other) -> bool:
        if type(self) is type(other):
            for field in self.fields.values():
                value = getattr(self, field.name)
                other_value = getattr(other, field.name)
                if value != other_value:
                    return False
            return True
        return False

    def __str__(self):
        return f'{self.__class__.__name__}({self.id})'

    def pprint(self) -> str:
        string = f'{self.__class__.__name__} {{'

        for field in self.fields.values():
            value = getattr(self, field.name)
            if value is None:
                continue

            if isinstance(field, ModelField):
                string += f'\n  {field.name}: {field.related_model.__name__}({value})'
            elif isinstance(field, ModelListField):
                string += f'\n  {field.name}: {field.related_model.__name__}{value}'
            else:
                string += f'\n  {field.name}: {repr(value)}'

        if string[-1] != '{':
            string += '\n'

        return string + '}'

    @classmethod
    def all_fields_odoo_names(cls):
        for field in cls.fields.values():
            for name in field.odoo_field_names:
                yield name

    @classmethod
    def field_odoo_names(cls, field_name):
        for name in cls.fields[field_name].odoo_field_names:
            yield name

    @classmethod
    def from_odoo(cls, **odoo_values) -> 'ModelBase':
        instance = cls()

        error_fields = []
        for field in instance.fields.values():
            try:
                field.from_odoo(instance, **odoo_values)
            except MissingValue:
                error_fields.append(field)

        if error_fields:
            missing_field_names = sorted(reduce(lambda s, f: s | f.odoo_field_names, error_fields, set()))
            raise MissingValues(instance, missing_field_names)

        return instance

    def save(self) -> None:
        values = {}
        for field in self.fields.values():
            if not field.has_changed(self):
                continue

            try:
                value = getattr(self, field.name)
                values.update(field.deconstruct(value))
            except ValueError:
                raise IncompleteModel(self, sorted(field.odoo_field_names))

        if 'id' in values:
            raise InvalidModelState('Instance id update is not supported')

        if self.id is None:
            self.id = connection.execute(self.Meta.name, 'create', values)
            ModelBase.id.save(self)
        elif values:
            connection.execute(self.Meta.name, 'write', self.id, values)

        for field in self.fields.values():
            if field.has_changed(self):
                field.save(self)

    def delete(self) -> None:
        connection.execute(self.Meta.name, 'unlink', [self.id])
        self.id = None
        self.id.save(self)


class Attachment(ModelBase):
    content = B64Field('datas')

    class Meta:
        name = 'ir.attachment'


class Model(ModelBase):

    @property
    def attachments(self) -> QuerySet:
        return Attachment.objects.filter(res_model=self.Meta.name, res_id=self.id)

    def render_report(self, report_name: str, **options) -> bytes:
        report_data = connection.render_report(report_name, self.id, **options)
        return b64decode(report_data['result'])

import re
import warnings
from base64 import b64decode
from datetime import date, datetime
from decimal import Decimal
from functools import cached_property, reduce
from typing import Any, Callable, Generic, Iterable, Optional, Type, TypeVar, Union
from zoneinfo import ZoneInfo

from odoo_orm.connection import OdooConnection
from odoo_orm.errors import (
    FieldDoesNotExist, IncompleteModel, InvalidModelState, LazyReferenceNotResolved, MissingField,
)

connection = OdooConnection.get_connection()

MB = TypeVar('MB', bound='ModelBase')
Rel = TypeVar('Rel', bound='ModelBase')
T = TypeVar('T')
Filter = tuple[str, str, Any]

c2s_pattern = re.compile(r'(?<!^)(?=[A-Z])')

ODOO_OPERATIONS = {
    'ne': '!=',
    'gt': '>',
    'ge': '>=',
    'lt': '<',
    'le': '<=',
    'in': 'in',
    'not_in': 'not in',
}


def c2s(s: str):
    return c2s_pattern.sub('.', s).lower()


class Field(Generic[T]):

    def __init__(self, *odoo_field_names: str, null=False) -> None:
        self.null = null
        self.odoo_field_names = set(odoo_field_names)

    def __set_name__(self, owner: Type[MB], name: str) -> None:
        self.model = owner
        self.name = name
        if not self.odoo_field_names:
            self.odoo_field_names = self.default_odoo_field_names
        self.initial_value_field_name = f'_field_{name}_initial'

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[T]:
        return getattr(instance, self.value_field_name)

    def __set__(self, instance: MB, value: Optional[T]) -> None:
        self.smart_set(instance, {self.name: value})

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        raise NotImplementedError

    @cached_property
    def value_field_name(self) -> str:
        return f'_field_{self.name}_value'

    @cached_property
    def assignable_field_name(self) -> str:
        return self.name

    def has_changed(self, instance: MB) -> bool:
        return getattr(instance, self.value_field_name) != getattr(instance, self.initial_value_field_name)

    def set_unchanged(self, instance: MB) -> None:
        setattr(instance, self.initial_value_field_name, getattr(instance, self.value_field_name))

    def smart_set(self, instance: MB, values: dict[str, Optional[T]], *, initial=False) -> None:
        value = values.get(self.assignable_field_name)
        setattr(instance, self.value_field_name, value)
        if initial:
            setattr(instance, self.initial_value_field_name, value)
        elif not hasattr(instance, self.initial_value_field_name):
            setattr(instance, self.initial_value_field_name, None)

    def construct(self, **values: Any) -> Any:
        raise NotImplementedError

    def construct_and_validate(self, instance: MB, **values: Any) -> Any:
        relevant_values = {k: v for k, v in values.items() if k in self.odoo_field_names}
        if len(relevant_values) < len(self.odoo_field_names):
            return None

        value = self.construct(**relevant_values)

        if value is None and not self.null:
            raise MissingField(instance, self.odoo_field_names)

        return value

    def deconstruct(self, value: Optional[T]) -> dict:
        raise NotImplementedError


class SimpleField(Generic[T], Field[T]):

    def __init__(self, odoo_field_name: str = None, /, *, null=False) -> None:
        if odoo_field_name is None:
            super().__init__(null=null)
        else:
            super().__init__(odoo_field_name, null=null)

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.name}

    @cached_property
    def odoo_field_name(self) -> str:
        return list(self.odoo_field_names)[0]

    def to_python(self, value: Any) -> Any:
        return value

    def construct(self, **values: Any) -> Any:
        value = values[self.odoo_field_name]
        if value is False:
            return None
        else:
            return self.to_python(value)

    def to_odoo(self, value: T) -> Any:
        return value

    def deconstruct(self, value: Optional[T]) -> dict:
        if value is not None:
            odoo_val = self.to_odoo(value)
        elif self.null:
            odoo_val = False
        else:
            raise ValueError('value cannot be None')

        return {self.odoo_field_name: odoo_val}


class IntegerField(SimpleField[int]):
    to_python = int

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[int]:
        return super().__get__(instance, owner)


class StringField(SimpleField[str]):

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[str]:
        return super().__get__(instance, owner)


class B64Field(SimpleField[bytes]):

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[bytes]:
        return super().__get__(instance, owner)

    def to_python(self, value: Any) -> bytes:
        return b64decode(value)


class BooleanField(SimpleField[bool]):

    def __get__(self, instance: MB, owner: Type[MB]) -> bool:
        return super().__get__(instance, owner)

    def construct(self, **values: Any) -> bool:
        # Do not check if null
        return bool(values[list(self.odoo_field_names)[0]])


class DecimalField(SimpleField[Decimal]):
    to_odoo = float

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[Decimal]:
        return super().__get__(instance, owner)

    def to_python(self, value: Any) -> Decimal:
        return Decimal(str(value)).quantize(Decimal('.01'))


class DateField(SimpleField[date]):
    date_format = '%Y-%m-%d'

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[date]:
        return super().__get__(instance, owner)

    def to_python(self, value: Any) -> date:
        return datetime.strptime(value, self.date_format).date()

    def to_odoo(self, value: date) -> Any:
        return date.strftime(value, self.date_format)


class DatetimeField(SimpleField[datetime]):
    datetime_format = '%Y-%m-%d %H:%M:%S'

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[datetime]:
        return super().__get__(instance, owner)

    def to_python(self, value: Any) -> datetime:
        return datetime.strptime(value, self.datetime_format).replace(tzinfo=ZoneInfo('UTC'))

    def to_odoo(self, value: datetime) -> Any:
        return datetime.strftime(value.astimezone(ZoneInfo('UTC')), self.datetime_format)


class LazyReference:
    field: 'RelatedField' = None
    model: 'ModelBase' = None

    def attach(self, field):
        self.field = field
        self._do_resolution()

    def resolve(self, model):
        self.model = model
        self._do_resolution()

    def _do_resolution(self):
        if self.field and self.model:
            self.field.related_model = self.model


def resolves(*refs: LazyReference):
    def wrapper(model: Model):
        for ref in refs:
            ref.resolve(model)
        return model

    return wrapper


class RelatedField(Generic[Rel, T], SimpleField[T]):

    def __init__(self, odoo_field_name: str = None, /, *, model: Union[Type[Rel], LazyReference], null=False) -> None:
        super().__init__(odoo_field_name, null=null)
        self.related_model = model

    def __set_name__(self, owner: Type[MB], name: str) -> None:
        super().__set_name__(owner, name)

        if self.related_model == 'self':
            self.related_model = owner
        elif isinstance(self.related_model, LazyReference):
            self.related_model.attach(self)
        elif not (isinstance(self.related_model, type) and issubclass(self.related_model, ModelBase)):
            raise Exception('Only subclasses of "ModelBase" and "self" are accepted as "model" argument of'
                            ' RelatedField')


class ModelField(Generic[Rel], RelatedField[Rel, Rel]):

    def __set_name__(self, owner: Type[MB], name: str) -> None:
        super().__set_name__(owner, name)

        self.instance_field_name = f'_field_{name}_instance'

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[Rel]:
        rel_id = getattr(instance, self.value_field_name)
        if rel_id is None:
            return None
        else:
            rel_instance = getattr(instance, self.instance_field_name, None)
            if rel_instance is None:
                rel_instance = self.related_model.objects.get(id=rel_id)
                setattr(instance, self.instance_field_name, rel_instance)
            return rel_instance

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.value_field_name}

    @cached_property
    def value_field_name(self) -> str:
        return f'{self.name}_id'

    @cached_property
    def assignable_field_name(self) -> str:
        return self.value_field_name

    def smart_set(self, instance: MB, values: dict[str, Optional[Rel]], *, initial=False) -> None:
        if self.name in values:
            value = values[self.name]
            setattr(instance, self.instance_field_name, value)
            _id_value = value and value.id
            super().smart_set(instance, {self.value_field_name: _id_value}, initial=initial)
        else:
            super().smart_set(instance, values, initial=initial)

    def to_python(self, value: Any) -> Any:
        return value[0]


class ModelListField(Generic[Rel], RelatedField[Rel, 'QuerySet[Rel]']):

    def __set_name__(self, owner: Type[MB], name: str) -> None:
        super().__set_name__(owner, name)

        self.queryset_field_name = f'_field_{name}_queryset'

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional['QuerySet[Rel]']:
        rel_ids = getattr(instance, self.value_field_name)
        if rel_ids is None:
            return None
        else:
            queryset = getattr(instance, self.queryset_field_name, None)
            if queryset is None:
                queryset = QuerySet(self.related_model).filter(id__in=rel_ids)
                setattr(instance, self.queryset_field_name, queryset)
            return queryset

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.value_field_name}

    @cached_property
    def value_field_name(self) -> str:
        return f'{self.name}_ids'

    @cached_property
    def assignable_field_name(self) -> str:
        return self.value_field_name

    def smart_set(self, instance: MB, values: dict[str, Optional[T]], *, initial=False) -> None:
        if self.name in values:
            value = values[self.name]
            ids = [i.id for i in value]

            queryset = QuerySet(self.related_model).filter(id__in=ids)
            queryset.cache = value
            setattr(instance, self.queryset_field_name, queryset)

            super().smart_set(instance, {self.value_field_name: ids}, initial=initial)
        else:
            super().smart_set(instance, values, initial=initial)

    def to_odoo(self, value: T) -> Any:
        return [(6, '_', value)]


class QuerySet(Generic[MB]):

    def __init__(self, model: Type[MB]) -> None:
        self.model = model
        self.cache: Optional[list[MB]] = None
        self.filters: list[Filter] = []
        self.options: dict[str, Any] = {}
        self.prefetches: set[str] = set()

    def __iter__(self) -> Iterable[MB]:
        if self.cache is None:
            self._execute()
        return iter(self.cache)

    def __getitem__(self, item: int) -> MB:
        if self.cache is None:
            self._execute()
        return self.cache[item]

    def __len__(self) -> int:
        if self.cache is None:
            self._execute()
        return len(self.cache)

    def __eq__(self, other) -> bool:
        if type(self) is type(other):
            return (self.filters == other.filters
                    and self.options == other.options
                    and (self.cache is None and other.cache is None and self.prefetches == other.prefetches
                         or self.cache is not None and self.cache == other.cache))
        return False

    def _execute(self) -> list[MB]:
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

        self.cache = []
        for data in res:
            for field in self.options['fields']:
                if field not in data:
                    raise FieldDoesNotExist(self.model, field)
            self.cache.append(self.model.from_odoo(**data))

        self._prefetch(*list(self.prefetches))

        return self.cache

    def _enhance(self, **kwargs) -> 'QuerySet[MB]':
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

    def filter(self, **kwargs) -> 'QuerySet[MB]':
        return self._enhance(filter=kwargs)

    def values(self, *fields: str) -> 'QuerySet[MB]':
        return self._enhance(values=fields)

    def limit(self, number: int) -> 'QuerySet[MB]':
        return self._enhance(limit=number)

    def order_by(self, *fields: str) -> 'QuerySet[MB]':
        return self._enhance(order_by=fields)

    def prefetch(self, *field_names: str) -> 'QuerySet[MB]':
        return self._enhance(prefetch=field_names)

    def get(self, **kwargs) -> MB:
        if kwargs:
            instances = self.filter(**kwargs)._execute()
        else:
            if self.cache is None:
                self._execute()
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
        followings_by_field_name = {}
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
                all_ids = set(getattr(instance, field.value_field_name) for instance in self
                              if getattr(instance, field.value_field_name) is not None)

                related = field.related_model.objects.filter(id__in=sorted(all_ids))
                if followings:
                    related = related.prefetch(*followings)

                for instance in self:
                    r_id = getattr(instance, field.value_field_name)
                    if r_id is None:
                        continue
                    rel = next(r for r in related if r.id == r_id)
                    setattr(instance, field_name, rel)
            elif isinstance(field, ModelListField):
                all_ids = set()
                for instance in self:
                    all_ids |= set(getattr(instance, field.value_field_name))

                related = field.related_model.objects.filter(id__in=sorted(all_ids))
                if followings:
                    related = related.prefetch(*followings)

                for instance in self:
                    r_ids = getattr(instance, field.value_field_name)
                    rels = [r for r in related if r.id in r_ids]
                    getattr(instance, field_name).cache = rels
            else:
                raise ValueError('Only support prefetch on RelatedFields')


class Manager(Generic[MB]):

    def __init__(self, model: Type[MB]) -> None:
        self.model = model

    def __getattr__(self, item: str) -> Callable[[], QuerySet[MB]]:
        queryset = self.get_queryset()

        if item == 'all':
            return lambda: queryset
        else:
            return getattr(queryset, item)

    def get_queryset(self) -> QuerySet[MB]:
        return QuerySet[MB](self.model)


class MetaModel(type):

    def __init__(cls: Type[MB], name: str, bases: tuple[type], attrs: dict[str, Any]) -> None:
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


class ModelBase(Generic[MB], metaclass=MetaModel):
    fields: dict[str, Field]
    objects: Manager[MB]

    id = IntegerField()

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

            field.smart_set(self, values)

    def __eq__(self, other) -> bool:
        if type(self) is type(other):
            for field in self.fields.values():
                value = getattr(self, field.value_field_name)
                other_value = getattr(other, field.value_field_name)
                if value != other_value:
                    return False
            return True
        return False

    def __str__(self):
        return f'{self.__class__.__name__}({self.id})'

    def pprint(self) -> str:
        string = f'{self.__class__.__name__} {{'

        for field in self.fields.values():
            value = getattr(self, field.value_field_name)
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
    def from_odoo(cls, **odoo_values) -> MB:
        instance = cls()

        error_fields = []
        for field in instance.fields.values():
            try:
                value = field.construct_and_validate(instance, **odoo_values)
                field.smart_set(instance, {field.assignable_field_name: value}, initial=True)
            except MissingField:
                error_fields.append(field)

        if error_fields:
            missing_field_names = sorted(reduce(lambda s, f: s | f.odoo_field_names, error_fields, set()))
            raise IncompleteModel(instance, missing_field_names)

        return instance

    def save(self) -> None:
        values = {}
        for field in self.fields.values():
            if not field.has_changed(self):
                continue

            try:
                value = getattr(self, field.value_field_name)
                values.update(field.deconstruct(value))
            except ValueError:
                message = (f'Model "{self}" is missing required field "{sorted(field.odoo_field_names)}".'
                           f' This will throw IncompleteModel exception in version 3.0.')
                warnings.warn(DeprecationWarning(message))
                values.update({name: False for name in field.odoo_field_names})

        if 'id' in values:
            raise InvalidModelState('Instance id update is not supported')

        if self.id is None:
            self.id = self._field_id_initial = connection.execute(self.Meta.name, 'create', values)
        elif values:
            connection.execute(self.Meta.name, 'write', self.id, values)

        for field in self.fields.values():
            if field.has_changed(self):
                field.set_unchanged(self)

    def delete(self) -> None:
        connection.execute(self.Meta.name, 'unlink', [self.id])
        self.id = self._field_id_initial = None


class Attachment(ModelBase['Attachment']):
    name = StringField()
    content = B64Field('datas')

    class Meta:
        name = 'ir.attachment'


class Model(Generic[MB], ModelBase[MB]):

    @property
    def attachments(self) -> QuerySet[Attachment]:
        return Attachment.objects.filter(res_model=self.Meta.name, res_id=self.id)

    def render_report(self, report_name: str, **options) -> bytes:
        report_data = connection.render_report(report_name, self.id, **options)
        return b64decode(report_data['result'])

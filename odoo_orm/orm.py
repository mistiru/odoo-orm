import re
from base64 import b64decode
from datetime import date, datetime
from decimal import Decimal
from functools import cached_property
from typing import Any, Generic, Iterable, Optional, Type, TypeVar

from odoo_orm.connection import OdooConnection
from odoo_orm.errors import MissingField

odoo = OdooConnection.get_connection()

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
        self.value_field_name = f'_field_{name}_value'
        # TODO check if available
        self.has_changed_field_name = f'_field_{name}_has_changed'
        # TODO check if available
        setattr(self.model, self.has_changed_field_name, False)

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[T]:
        return getattr(instance, self.value_field_name, None)

    def __set__(self, instance: MB, value: Optional[T]) -> None:
        if value != getattr(instance, self.value_field_name, None):
            setattr(instance, self.has_changed_field_name, True)
        setattr(instance, self.value_field_name, value)

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.name}

    def construct(self, **values: Any) -> Any:
        raise NotImplementedError

    def construct_and_validate(self, instance: MB, **values: Any) -> Any:
        relevant_values = {k: v for k, v in values.items() if k in self.odoo_field_names}
        if len(relevant_values) < len(self.odoo_field_names):
            return None

        value = self.construct(**relevant_values)

        if value is None and not self.null:
            if isinstance(self, SimpleField):
                raise MissingField(f'Sur Odoo, {instance.user_friendly_display} ne possède pas de valeur pour son'
                                   f" champ '{self.odoo_field_name}'")
            else:
                field_names = "', '".join(self.odoo_field_names)
                raise MissingField(f'Sur Odoo, {instance.user_friendly_display} ne possède pas de valeur pour au'
                                   f" moins un des champs '{field_names}'")

        return value

    def deconstruct(self, value: T) -> dict:
        raise NotImplementedError


class SimpleField(Generic[T], Field[T]):

    def __init__(self, odoo_field_name: str = None, /, *, null=False) -> None:
        if odoo_field_name is None:
            super().__init__(null=null)
        else:
            super().__init__(odoo_field_name, null=null)

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

    def deconstruct(self, value: T) -> dict:
        return {self.odoo_field_name: self.to_odoo(value)}


class IntegerField(SimpleField[int]):
    to_python = int

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[int]:
        return super().__get__(instance, owner)


class StringField(SimpleField[str]):

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[str]:
        return super().__get__(instance, owner)


class B64Field(SimpleField[bytes]):
    to_python = b64decode

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[bytes]:
        return super().__get__(instance, owner)


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

    def __init__(self, odoo_field_name: str = None, /, *, date_format: str, null=False) -> None:
        super().__init__(odoo_field_name, null=null)
        self.format = date_format

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[date]:
        return super().__get__(instance, owner)

    def to_python(self, value: Any) -> date:
        return datetime.strptime(value, self.format).date()

    def to_odoo(self, value: date) -> Any:
        return date.strftime(value, self.format)


class RelatedField(Generic[Rel, T], SimpleField[T]):

    def __init__(self, odoo_field_name: str = None, /, *, model: Type[Rel], null=False) -> None:
        super().__init__(odoo_field_name, null=null)
        self.related_model = model


class ModelField(Generic[Rel], RelatedField[Rel, Rel]):

    def __set_name__(self, owner: Type[MB], name: str) -> None:
        self.id_field_name = f'{name}_id'
        super().__set_name__(owner, name)
        # TODO check if available
        val = cached_property(self.get_related_or_none)
        val.__set_name__(owner, self.value_field_name)
        setattr(owner, self.value_field_name, val)

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional[Rel]:
        return super().__get__(instance, owner)

    def __set__(self, instance: MB, value: Optional[Rel]) -> None:
        if instance.id != getattr(instance, self.id_field_name):
            setattr(instance, self.has_changed_field_name, True)
        setattr(instance, self.value_field_name, value)
        setattr(instance, self.id_field_name, value and value.id or None)

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.id_field_name}

    def to_python(self, value: Any) -> Any:
        return value[0]

    def get_related_or_none(self, instance: MB) -> Optional[Rel]:
        related_model_id = getattr(instance, self.id_field_name)
        if related_model_id is None:
            return None
        else:
            return self.related_model.objects.get(id=related_model_id)


class ModelListField(Generic[Rel], RelatedField[Rel, 'RelatedManager[Rel]']):

    def __set_name__(self, owner: Type[MB], name: str) -> None:
        self.ids_field_name = f'{name}_ids'
        super().__set_name__(owner, name)
        # TODO check if available
        val = cached_property(self.get_related_manager_or_none)
        val.__set_name__(owner, self.value_field_name)
        setattr(owner, self.value_field_name, val)

    def __get__(self, instance: MB, owner: Type[MB]) -> Optional['RelatedManager[Rel]']:
        return super().__get__(instance, owner)

    def __set__(self, instance: MB, value: Optional['RelatedManager[Rel]']) -> None:
        raise AttributeError

    @cached_property
    def default_odoo_field_names(self) -> set[str]:
        return {self.ids_field_name}

    def get_related_manager_or_none(self, instance: MB) -> Optional['RelatedManager[Rel]']:
        related_model_ids = getattr(instance, self.ids_field_name)
        if related_model_ids is None:
            return None
        else:
            return RelatedManager(self.related_model, instance, self)


class MetaModel(type):

    def __init__(cls: Type[MB], name: str, bases: tuple[type], attrs: dict[str, Any]) -> None:
        super().__init__(name, bases, attrs)

        cls.fields = {}
        if name != 'Model':
            for base in reversed(bases):
                if issubclass(base, ModelBase):
                    for attr_name, field in base.fields.items():
                        cls.fields[attr_name] = field
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field):
                cls.fields[attr_name] = attr_value

        if name != 'Model' and getattr(cls, 'objects', None) is None:
            cls.objects = Manager(cls)

        if name != 'Model':
            if getattr(cls, 'Meta', None) is None:
                cls.Meta = type('Meta', (), {})
            if getattr(cls.Meta, 'name', None) is None:
                cls.Meta.name = c2s(name)


class ModelBase(Generic[MB], metaclass=MetaModel):
    fields: dict[str, Field]
    objects: 'Manager[MB]'

    id = IntegerField()

    class Meta:
        name: str

    class DoesNotExist(Exception):
        pass

    class MultipleObjectsReturned(Exception):
        pass

    def __init__(self, **odoo_dict: Any) -> None:
        for field in self.fields.values():
            value = field.construct_and_validate(self, **odoo_dict)
            if isinstance(field, ModelField):
                setattr(self, field.id_field_name, value)
            elif isinstance(field, ModelListField):
                setattr(self, field.ids_field_name, value)
            else:
                setattr(self, field.value_field_name, value)

    def __repr__(self) -> str:
        return self.pprint()

    def pprint(self, padding=0) -> str:
        string = f'{self.__class__.__name__} {{\n'

        for field in self.fields.values():
            value = getattr(self, field.name)
            if value is None:
                continue

            if isinstance(field, ModelField):
                string += f'{" " * 2 * padding}  {field.name}: {value.pprint(padding + 1)}\n'
            elif isinstance(field, ModelListField):
                string += f'{" " * 2 * padding}  {field.name}: [\n'
                string += ',\n'.join(f'{" " * 2 * padding}    {instance.pprint(padding + 2)}'
                                     for instance in value)
                string += ']\n'
            else:
                string += f'{" " * 2 * padding}  {field.name}: {repr(value)}\n'

        return f'{string}{" " * 2 * padding}}}'

    @classmethod
    def all_fields_odoo_names(cls):
        for field in cls.fields.values():
            for name in field.odoo_field_names:
                yield name

    @classmethod
    def field_odoo_names(cls, field_name):
        for name in cls.fields[field_name].odoo_field_names:
            yield name

    def save(self) -> None:
        values = {}
        for field in self.fields.values():
            if isinstance(field, ModelListField):
                continue

            if isinstance(field, ModelField):
                value = getattr(self, field.id_field_name, None)
            else:
                value = getattr(self, field.value_field_name, None)

            if value is None:
                continue

            has_changed = getattr(self, field.has_changed_field_name)
            if not has_changed:
                continue

            values.update(field.deconstruct(value))

        odoo.execute(self.Meta.name, 'write', self.id, values)


class Attachment(ModelBase['Attachment']):
    content = B64Field('datas')

    class Meta:
        name = 'ir.attachment'


class Model(ModelBase['Model']):

    @property
    def attachments(self) -> 'QuerySet[Attachment]':
        return Attachment.objects.values('id').filter(res_model=self.Meta.name, res_id=self.id)

    def render_report(self, report_name: str, **options) -> bytes:
        report_data = odoo.render_report(report_name, self.id, **options)
        return b64decode(report_data['result'])


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

    def _execute(self) -> list[MB]:
        if 'fields' not in self.options:
            self.options['fields'] = list(self.model.all_fields_odoo_names())

        res = odoo.execute(self.model.Meta.name, 'search_read', self.filters, **self.options)

        self.cache = [self.model(**data) for data in res]

        self._prefetch(*list(self.prefetches))

        return self.cache

    def _enhance(self, **kwargs) -> 'QuerySet[MB]':
        new = QuerySet(self.model)
        new.filters = self.filters.copy()
        new.options = self.options.copy()

        for kw, val in kwargs.items():
            if kw == 'limit':
                new.options['limit'] = 1
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
            else:
                raise Exception(f'Argument not recognized {kw}')

        return new

    def filter(self, **kwargs) -> 'QuerySet[MB]':
        return self._enhance(filter=kwargs)

    def values(self, *fields: str) -> 'QuerySet[MB]':
        return self._enhance(values=fields)

    def limit(self, number: int) -> 'QuerySet[MB]':
        return self._enhance(limit=number)

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

    def prefetch(self, *field_names: str) -> 'QuerySet[MB]':
        if self.cache is None:
            self.prefetches |= set(field_names)
        else:
            self._prefetch(*field_names)
        return self

    def _prefetch(self, *field_names: str) -> None:
        for field_name in field_names:
            # TODO manage '__' (eg: invoice.pickings.packages.lots -> packages__lots)
            field = self.model.fields[field_name]

            if isinstance(field, ModelField):
                all_ids = set(getattr(instance, field.id_field_name) for instance in self)
                related = field.related_model.objects.filter(id__in=list(all_ids))
                for instance in self:
                    setattr(instance, field_name,
                            next(r for r in related if r.id == getattr(instance, field.id_field_name)))
            elif isinstance(field, ModelListField):
                all_ids = set()
                for instance in self:
                    all_ids |= set(getattr(instance, field.ids_field_name))

                related = field.related_model.objects.filter(id__in=list(all_ids))

                for instance in self:
                    selection = [r for r in related if r.id in getattr(instance, field.ids_field_name)]
                    getattr(instance, field_name).queryset.cache = selection
            else:
                raise ValueError('Only support prefetch on RelatedFields')


class Manager(Generic[MB]):

    def __init__(self, model: Type[MB]) -> None:
        self.queryset = QuerySet[MB](model)

    def all(self) -> QuerySet[MB]:
        return self.queryset

    def filter(self, **kwargs) -> QuerySet[MB]:
        return self.queryset.filter(**kwargs)

    def values(self, *fields: str) -> QuerySet[MB]:
        return self.queryset.values(*fields)

    def limit(self, number: int) -> QuerySet[MB]:
        return self.queryset.limit(number)

    def get(self, **kwargs) -> MB:
        return self.queryset.get(**kwargs)


class RelatedManager(Generic[Rel], Manager[Rel]):

    def __init__(self, model: Type[MB], instance: MB, field: ModelListField[Rel]) -> None:
        super().__init__(model)
        self.instance = instance
        self.field = field
        self.queryset = self.queryset.filter(id__in=getattr(instance, field.ids_field_name))

    def __iter__(self) -> Iterable[Rel]:
        return iter(self.queryset)

    def __getitem__(self, item: int) -> Rel:
        return self.queryset[item]

    def __len__(self) -> int:
        return len(self.queryset)

    def set(self, instances: list[Rel]) -> None:
        ids: list[int] = [i.id for i in instances]
        self.field: ModelListField[Rel]
        setattr(self.instance, self.field.ids_field_name, ids)
        self.queryset = QuerySet(self.queryset.model).filter(id__in=ids)
        self.queryset.cache = instances
        (odoo.execute(self.instance.Meta.name, 'write', self.instance.id,
                      {list(self.field.odoo_field_names)[0]: [(6, '_', ids)]}))

    def prefetch(self, *field_names: str) -> QuerySet[Rel]:
        return self.queryset.prefetch(*field_names)

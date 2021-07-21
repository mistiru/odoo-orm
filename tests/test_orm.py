from _zoneinfo import ZoneInfo
from base64 import b64encode
from datetime import date, datetime
from operator import attrgetter
from unittest.mock import call, MagicMock

import pytest

from odoo_orm.errors import FieldDoesNotExist, IncompleteModel, InvalidModelState
from odoo_orm.orm import (Attachment, B64Field, BooleanField, c2s, DateField, DatetimeField, DecimalField, IntegerField,
                          LazyReference, Manager, Model, ModelBase, ModelField, ModelListField, QuerySet, resolves,
                          StringField)


def test_camel_to_snake_case():
    assert c2s('Test') == 'test'
    assert c2s('') == ''
    assert c2s('test') == 'test'
    assert c2s('TestTest') == 'test.test'
    assert c2s('testTest') == 'test.test'
    assert c2s('TesT') == 'tes.t'
    assert c2s('TTT') == 't.t.t'
    assert c2s('T') == 't'


late_declared_model = LazyReference()


class SomeModel(ModelBase['SomeModel']):
    some_field = StringField()
    some_related_field = ModelField(model=ModelBase)
    some_list_field = ModelListField(model=ModelBase)
    some_named_field = StringField('named_string')
    some_null_field = StringField(null=True)
    some_b64_field = B64Field()
    some_boolean_field = BooleanField()
    some_decimal_field = DecimalField()
    some_date_field = DateField()
    some_chain_field = ModelField(model='self')
    some_chain_list_field = ModelListField(model='self')
    some_nullable_related_field = ModelField(model=ModelBase, null=True)
    some_datetime_field = DatetimeField()
    some_resolvable_field = ModelField(model=late_declared_model)

    ODOO_DEFAULT_VALUES_DICT = {
        'some_field': '',
        'some_related_field_id': [999999, ''],
        'some_list_field_ids': [],
        'named_string': '',
        'some_null_field': False,
        'some_b64_field': '',
        'some_boolean_field': False,
        'some_decimal_field': 0.0,
        'some_date_field': datetime.strftime(datetime(1999, 1, 1), DateField.date_format),
        'some_chain_field_id': [999999, ''],
        'some_chain_list_field_ids': [],
        'some_nullable_related_field_id': False,
        'some_datetime_field': datetime.strftime(datetime(1999, 1, 1), DatetimeField.datetime_format),
        'some_resolvable_field_id': [999999, ''],
    }

    @staticmethod
    def populate_odoo_return_values(*odoo_return_values_list: list[dict]) -> tuple[list[dict]]:
        for odoo_return_values in odoo_return_values_list:
            for odoo_return_value in odoo_return_values:
                for field, default_value in SomeModel.ODOO_DEFAULT_VALUES_DICT.items():
                    odoo_return_value.setdefault(field, default_value)

        return odoo_return_values_list


@resolves(late_declared_model)
class LateDeclaredModel(ModelBase['LateDeclaredModel']):
    some_model = ModelField(model=SomeModel)


@pytest.fixture
def basic_instance():
    return SomeModel.from_odoo(id=1, some_field='tut', some_related_field_id=[2, 'tut'], some_list_field_ids=[3, 4],
                               named_string='pouet')


# class MoneyField(Field[tuple[int, str]]):
#
#     def default_odoo_field_names(self) -> set[str]:
#         return {f'{self.name}_amount', f'{self.name}_currency'}
#
#     def construct(self, **values: Any) -> Any:
#         return (values[field_name] for field_name in self.odoo_field_names)
#         value = values[self.odoo_field_name]
#         if value is False:
#             return None
#         else:
#             return self.to_python(value)
#
#     def deconstruct(self, value: T) -> dict:
#         return {self.odoo_field_name: self.to_odoo(value)}


class TestMetaModel:

    def test_model_fields(self):
        for name, field_type in (('id', IntegerField), ('some_field', StringField), ('some_related_field', ModelField),
                                 ('some_list_field', ModelListField), ('some_named_field', StringField),
                                 ('some_null_field', StringField), ('some_b64_field', B64Field),
                                 ('some_boolean_field', BooleanField), ('some_decimal_field', DecimalField),
                                 ('some_date_field', DateField), ('some_chain_field', ModelField),
                                 ('some_chain_list_field', ModelListField), ('some_nullable_related_field', ModelField),
                                 ('some_datetime_field', DatetimeField), ('some_resolvable_field', ModelField)):
            assert name in SomeModel.fields
            assert isinstance(SomeModel.fields[name], field_type)

    def test_model_objects(self):
        assert hasattr(SomeModel, 'objects')
        assert isinstance(SomeModel.objects, Manager)
        assert SomeModel.objects.queryset.model == SomeModel

    def test_model_meta(self):
        assert hasattr(SomeModel, 'Meta')
        assert isinstance(SomeModel.Meta, type)
        assert hasattr(SomeModel.Meta, 'name')
        assert SomeModel.Meta.name == 'some.model'

    def test_model_custom_meta_name(self):
        class EmptyModel(ModelBase):
            class Meta:
                name = 'tut'

        assert EmptyModel.Meta.name == 'tut'


class TestField:

    def test_disabled_fields_are_none(self):
        instance = SomeModel()
        assert instance.id is None
        assert instance.some_field is None
        assert instance.some_related_field is None
        assert instance.some_list_field is None
        assert instance.some_named_field is None

    def test_odoo_missing_fields(self):
        with pytest.raises(IncompleteModel):
            SomeModel.from_odoo(id=False)

    def test_odoo_missing_field_is_ok_if_null(self):
        instance = SomeModel.from_odoo(some_null_field=False)
        assert instance.some_null_field is None

    def test_odoo_b64field_decodes(self):
        instance = SomeModel.from_odoo(some_b64_field=b64encode(b'tutturu'))
        assert instance.some_b64_field == b'tutturu'

    def test_odoo_false_boolean_field_is_ok(self):
        instance = SomeModel.from_odoo(some_boolean_field=False)
        assert instance.some_boolean_field is False

    def test_decimal_field_has_two_decimals(self):
        for raw, string in ((3, '3.00'), (3.1, '3.10'), (3.1415, '3.14')):
            instance = SomeModel.from_odoo(some_decimal_field=raw)
            assert str(instance.some_decimal_field) == string

    def test_date_field(self):
        date_str = '2020-12-17'
        instance = SomeModel.from_odoo(some_date_field=date_str)
        assert instance.some_date_field == date(2020, 12, 17)
        assert (instance.fields['some_date_field']
                .deconstruct(instance.some_date_field)) == {'some_date_field': date_str}

    def test_datetime_field(self):
        datetime_str = '2020-12-17 14:37:35'
        instance = SomeModel.from_odoo(some_datetime_field=datetime_str)
        assert instance.some_datetime_field == datetime(2020, 12, 17, 14, 37, 35, tzinfo=ZoneInfo('UTC'))
        assert (instance.fields['some_datetime_field']
                .deconstruct(instance.some_datetime_field)) == {'some_datetime_field': datetime_str}

    @pytest.mark.connection_returns([{'id': 2}])
    def test_model_field_get_from_id(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(some_related_field_id=[2, 'tut'])
        assert instance.some_related_field.id == 2
        spy_execute.assert_called_once_with('model.base', 'read', [2], fields=['id'])

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 2}]))
    def test_chain_field(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(some_chain_field_id=[2, 'pouet'])
        assert instance.some_chain_field.id == 2
        spy_execute.assert_called_once_with('some.model', 'read', [2], fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([{'id': 3}, {'id': 4}])
    def test_model_list_field_get_from_ids(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(some_list_field_ids=[3, 4])
        assert list(map(attrgetter('id'), instance.some_list_field)) == [3, 4]
        spy_execute.assert_called_once_with('model.base', 'read', [3, 4], fields=['id'])

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 3}, {'id': 4}]))
    def test_chain_list_field(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(some_chain_list_field_ids=[3, 4])
        assert list(map(attrgetter('id'), instance.some_chain_list_field)) == [3, 4]
        spy_execute.assert_called_once_with('some.model', 'read', [3, 4],
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([{'id': 1, 'some_model_id': [2, 'ohayou']}])
    def test_resolvable_field(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(some_resolvable_field_id=[1, 'tutturu'])
        assert instance.some_resolvable_field.id == 1
        spy_execute.assert_called_once_with('late.declared.model', 'read', [1],
                                            fields=list(LateDeclaredModel.all_fields_odoo_names()))


class TestQuerySet:

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 42}]))
    def test_can_iter(self, spy_execute: MagicMock):
        [instance] = list(QuerySet(SomeModel))
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=list(SomeModel.all_fields_odoo_names()))
        assert instance.id == 42

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 42}, {'id': 69}]))
    def test_can_get_item(self, spy_execute: MagicMock):
        instance = QuerySet(SomeModel)[1]
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=list(SomeModel.all_fields_odoo_names()))
        assert instance.id == 69

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 42}, {'id': 69}]))
    def test_can_get_length(self, spy_execute: MagicMock):
        length = len(QuerySet(SomeModel))
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=list(SomeModel.all_fields_odoo_names()))
        assert length == 2

    @pytest.mark.connection_returns([{'id': 1}])
    def test_has_cache(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel).values()
        spy_execute.assert_not_called()
        assert len(queryset) == 1
        assert len(queryset) == 1
        assert queryset.get().id == 1
        assert list(queryset) == [SomeModel(id=1)]
        assert queryset[0].id == 1
        spy_execute.assert_called_once()

    @pytest.mark.connection_returns([])
    def test_filter_simple(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).filter(id=1))
        spy_execute.assert_called_once_with('some.model', 'read', [1], fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([])
    def test_filter_multiple(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).filter(id=1).filter(tut=2))
        spy_execute.assert_called_once_with('some.model', 'search_read', [('id', '=', 1), ('tut', '=', 2)],
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([])
    def test_filter_double(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).filter(id=1, tut=2))
        spy_execute.assert_called_once_with('some.model', 'search_read', [('id', '=', 1), ('tut', '=', 2)],
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([])
    def test_filter_related(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).filter(id=1, tut__pouet=2))
        spy_execute.assert_called_once_with('some.model', 'search_read', [('id', '=', 1), ('tut.pouet', '=', 2)],
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([], [], [], [], [], [], [])
    def test_filter_(self, spy_execute: MagicMock):
        tests = (('in', 'in'), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<='), ('ne', '!='), ('not_in', 'not in'))
        for python, odoo in tests:
            len(QuerySet(SomeModel).filter(id=1, **{f'tut__pouet__{python}': 2}))
            spy_execute.assert_called_once_with('some.model', 'search_read', [('id', '=', 1), ('tut.pouet', odoo, 2)],
                                                fields=list(SomeModel.all_fields_odoo_names()))
            spy_execute.reset_mock()

    @pytest.mark.connection_returns([])
    def test_values(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).values('id', 'some_named_field'))
        spy_execute.assert_called_once_with('some.model', 'search_read', [], fields=['id', 'named_string'])

        with pytest.raises(KeyError):
            len(QuerySet(SomeModel).values('id', 'pouet'))

    @pytest.mark.connection_returns([{}])
    def test_fail_on_missing_value(self, spy_execute: MagicMock):
        with pytest.raises(FieldDoesNotExist):
            len(QuerySet(SomeModel))
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([{'id': 1, 'some_field': 'pop'}],
                                    [{'id': 2}])
    def test_fail_on_missing_values_respects_values(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).values('some_field'))
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=['some_field'])

        spy_execute.reset_mock()

        with pytest.raises(FieldDoesNotExist):
            len(QuerySet(SomeModel).values('some_field'))
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=['some_field'])

    @pytest.mark.connection_returns([])
    def test_limit(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).limit(7))
        spy_execute.assert_called_once_with('some.model', 'search_read', [], limit=7,
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([])
    def test_order_by(self, spy_execute: MagicMock):
        len(QuerySet(SomeModel).order_by('id', '-some_named_field'))
        spy_execute.assert_called_once_with('some.model', 'search_read', [], order='id asc, some_named_field desc',
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 1}]))
    def test_get(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel).filter(id=1)
        assert queryset.get().id == 1
        spy_execute.assert_called_once_with('some.model', 'read', [1], fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 1}]))
    def test_get_accepts_arguments(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel)
        assert queryset.get(id=1).id == 1
        spy_execute.assert_called_once_with('some.model', 'read', [1], fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns([])
    def test_get_does_not_exist(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel)
        with pytest.raises(SomeModel.DoesNotExist):
            queryset.get(id=0)
        spy_execute.assert_called_once_with('some.model', 'read', [0], fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 1}, {'id': 2}]))
    def test_get_multiple(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel)
        with pytest.raises(SomeModel.MultipleObjectsReturned):
            queryset.get(id__le=2)
        spy_execute.assert_called_once_with('some.model', 'search_read', [('id', '<=', 2)],
                                            fields=list(SomeModel.all_fields_odoo_names()))

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_related_field_id': [2, 'tut']},
                                                {'id': 2, 'some_related_field_id': [2, 'tut']},
                                                {'id': 3, 'some_related_field_id': [4, 'pouet']}],
                                               [{'id': 2}, {'id': 4}]))
    def test_prefetch(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel).prefetch('some_related_field')
        spy_execute.assert_not_called()
        instances = list(queryset)
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [2, 4], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_related_field.id == 2
        assert instances[1].some_related_field.id == 2
        assert instances[2].some_related_field.id == 4
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_related_field_id': [2, 'tat']},
                                                {'id': 2, 'some_related_field_id': [2, 'tat']},
                                                {'id': 3, 'some_related_field_id': [4, 'pouet']}],
                                               [{'id': 2}, {'id': 4}]))
    def test_prefetch_cached_queryset(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel)
        instances = list(queryset)
        spy_execute.assert_called_once_with('some.model', 'search_read', [],
                                            fields=list(SomeModel.all_fields_odoo_names()))
        spy_execute.reset_mock()

        queryset.prefetch('some_related_field')
        spy_execute.assert_called_once_with('model.base', 'read', [2, 4],
                                            fields=list(ModelBase.all_fields_odoo_names()))
        spy_execute.reset_mock()
        assert instances[0].some_related_field.id == 2
        assert instances[1].some_related_field.id == 2
        assert instances[2].some_related_field.id == 4
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(*SomeModel.populate_odoo_return_values([{'id': 1}, {'id': 2}]))
    def test_prefetch_requires_real_field(self, spy_execute: MagicMock):
        with pytest.raises(KeyError):
            list(QuerySet(SomeModel).prefetch('tut'))

    @pytest.mark.connection_returns([])
    def test_do_not_prefetch_if_queryset_is_empty(self, spy_execute: MagicMock):
        list(QuerySet(SomeModel).prefetch('some_related_field'))
        spy_execute.assert_called_once()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_list_field_ids': [2, 3]},
                                                {'id': 2, 'some_list_field_ids': [2]},
                                                {'id': 3, 'some_list_field_ids': [4]}],
                                               [{'id': 2}, {'id': 3}, {'id': 4}]))
    def test_prefetch_on_model_list(self, spy_execute: MagicMock):
        queryset = QuerySet(SomeModel).prefetch('some_list_field')
        spy_execute.assert_not_called()
        instances = list(queryset)
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [2, 3, 4], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_list_field.cache == [ModelBase(id=2), ModelBase(id=3)]
        assert instances[1].some_list_field.cache == [ModelBase(id=2)]
        assert instances[2].some_list_field.cache == [ModelBase(id=4)]
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_chain_field_id': [3, 'tut']},
                                                {'id': 2, 'some_chain_field_id': [4, 'pouet']}],
                                               [{'id': 3, 'some_related_field_id': [5, 'bidule']},
                                                {'id': 4, 'some_related_field_id': [5, 'bidule']}],
                                               [{'id': 5}]))
    def test_prefetch_follows_related_fields(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel).prefetch('some_chain_field__some_related_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [5], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_field.id == 3
        assert instances[1].some_chain_field.id == 4
        assert instances[0].some_chain_field.some_related_field.id == 5
        assert instances[1].some_chain_field.some_related_field.id == 5
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_related_field_id': [3, 'tut'], 'some_list_field_ids': [5, 6]},
             {'id': 2, 'some_related_field_id': [4, 'pouet'], 'some_list_field_ids': [6, 7]}],
            [{'id': 5}, {'id': 6}, {'id': 7}],
            [{'id': 3}, {'id': 4}]))
    def test_prefetch_multiple_fields(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel).prefetch('some_related_field', 'some_list_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [5, 6, 7], fields=list(ModelBase.all_fields_odoo_names())),
            call('model.base', 'read', [3, 4], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_related_field.id == 3
        assert instances[1].some_related_field.id == 4
        assert instances[0].some_list_field[0].id == 5
        assert instances[0].some_list_field[1].id == 6
        assert instances[1].some_list_field[0].id == 6
        assert instances[1].some_list_field[1].id == 7
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_nullable_related_field_id': False},
                                                {'id': 2, 'some_nullable_related_field_id': [3, 'tut']}],
                                               [{'id': 3}]))
    def test_prefetch_ignore_none_values(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel).prefetch('some_nullable_related_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [3], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_nullable_related_field is None
        assert instances[1].some_nullable_related_field.id == 3
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_related_field_id': [2, 'pouet']}],
                                               [{'id': 2}]))
    def test_prefetch_not_last_operation_1(self, spy_execute: MagicMock):
        instance = QuerySet(SomeModel).prefetch('some_related_field').get()
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [2], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instance.some_related_field.id == 2
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_related_field_id': [2, 'prout']}],
                                               [{'id': 2}]))
    def test_prefetch_not_last_operation_2(self, spy_execute: MagicMock):
        instance = QuerySet(SomeModel).prefetch('some_related_field').get(some_field='pouet')
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [('some_field', '=', 'pouet')],
                 fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [2], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instance.some_related_field.id == 2
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_list_field_ids': [3, 4]},
                                                {'id': 2, 'some_list_field_ids': [4, 5]}],
                                               [{'id': 3}, {'id': 4}, {'id': 5}]))
    def test_prefetch_not_last_operation_3(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel).prefetch('some_list_field').filter(some_field='pouet'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [('some_field', '=', 'pouet')],
                 fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [3, 4, 5], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_list_field[0].id == 3
        assert instances[0].some_list_field[1].id == 4
        assert instances[1].some_list_field[0].id == 4
        assert instances[1].some_list_field[1].id == 5
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values([{'id': 1, 'some_chain_field_id': [2, 'tet']}],
                                               [{'id': 2, 'some_chain_field_id': [3, 'tat']}],
                                               [{'id': 3}]))
    def test_prefetch_recursive(self, spy_execute: MagicMock):
        instance = QuerySet(SomeModel).prefetch('some_chain_field__some_chain_field').get()
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [2], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3], fields=list(SomeModel.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instance.some_chain_field.id == 2
        assert instance.some_chain_field.some_chain_field.id == 3
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_field_id': [2, 'tet'], 'some_related_field_id': [3, 'tot']}],
            [{'id': 2}],
            [{'id': 3}]))
    def test_prefetch_alternate(self, spy_execute: MagicMock):
        instance = (QuerySet(SomeModel).prefetch('some_chain_field').filter(some_field='tyt')
                    .prefetch('some_related_field').get())
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [('some_field', '=', 'tyt')],
                 fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [2], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [3], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instance.some_chain_field.id == 2
        assert instance.some_related_field.id == 3
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_field_id': [3, '']},
             {'id': 2, 'some_chain_field_id': [4, '']}],
            [{'id': 3, 'some_related_field_id': [5, ''], 'some_chain_field_id': [6, '']},
             {'id': 4, 'some_related_field_id': [7, ''], 'some_chain_field_id': [8, '']}],
            [{'id': 6}, {'id': 8}],
            [{'id': 5}, {'id': 7}]))
    def test_multiple_prefetches_same_model_0(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_field__some_related_field',
                                   'some_chain_field__some_chain_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [6, 8], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [5, 7], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_field.id == 3
        assert instances[0].some_chain_field.some_related_field.id == 5
        assert instances[0].some_chain_field.some_chain_field.id == 6
        assert instances[1].some_chain_field.id == 4
        assert instances[1].some_chain_field.some_related_field.id == 7
        assert instances[1].some_chain_field.some_chain_field.id == 8
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_field_id': [3, '']},
             {'id': 2, 'some_chain_field_id': [3, '']}],
            [{'id': 3, 'some_related_field_id': [4, ''], 'some_chain_field_id': [5, '']}],
            [{'id': 5}],
            [{'id': 4}]))
    def test_multiple_prefetches_same_model_1(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_field__some_related_field',
                                   'some_chain_field__some_chain_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [5], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [4], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_field.id == 3
        assert instances[0].some_chain_field.some_related_field.id == 4
        assert instances[0].some_chain_field.some_chain_field.id == 5
        assert instances[1].some_chain_field.id == 3
        assert instances[1].some_chain_field.some_related_field.id == 4
        assert instances[1].some_chain_field.some_chain_field.id == 5
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_field_id': [3, '']},
             {'id': 2, 'some_chain_field_id': [4, '']}],
            [{'id': 3, 'some_list_field_ids': [5, 6], 'some_chain_list_field_ids': [7, 8]},
             {'id': 4, 'some_list_field_ids': [9, 10], 'some_chain_list_field_ids': [11, 12]}],
            [{'id': 7}, {'id': 8}, {'id': 11}, {'id': 12}],
            [{'id': 5}, {'id': 6}, {'id': 9}, {'id': 10}]))
    def test_multiple_prefetches_same_model_2(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_field__some_list_field',
                                   'some_chain_field__some_chain_list_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [7, 8, 11, 12], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [5, 6, 9, 10], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_field.id == 3
        assert instances[0].some_chain_field.some_list_field[0].id == 5
        assert instances[0].some_chain_field.some_list_field[1].id == 6
        assert instances[0].some_chain_field.some_chain_list_field[0].id == 7
        assert instances[0].some_chain_field.some_chain_list_field[1].id == 8
        assert instances[1].some_chain_field.id == 4
        assert instances[1].some_chain_field.some_list_field[0].id == 9
        assert instances[1].some_chain_field.some_list_field[1].id == 10
        assert instances[1].some_chain_field.some_chain_list_field[0].id == 11
        assert instances[1].some_chain_field.some_chain_list_field[1].id == 12
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_field_id': [3, '']},
             {'id': 2, 'some_chain_field_id': [4, '']}],
            [{'id': 3, 'some_related_field_id': [5, ''], 'some_list_field_ids': [6, 7]},
             {'id': 4, 'some_related_field_id': [8, ''], 'some_list_field_ids': [9, 10]}],
            [{'id': 6}, {'id': 7}, {'id': 9}, {'id': 10}],
            [{'id': 5}, {'id': 8}]))
    def test_multiple_prefetches_same_model_3(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_field__some_related_field',
                                   'some_chain_field__some_list_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [6, 7, 9, 10], fields=list(ModelBase.all_fields_odoo_names())),
            call('model.base', 'read', [5, 8], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_field.id == 3
        assert instances[0].some_chain_field.some_related_field.id == 5
        assert instances[0].some_chain_field.some_list_field[0].id == 6
        assert instances[0].some_chain_field.some_list_field[1].id == 7
        assert instances[1].some_chain_field.id == 4
        assert instances[1].some_chain_field.some_related_field.id == 8
        assert instances[1].some_chain_field.some_list_field[0].id == 9
        assert instances[1].some_chain_field.some_list_field[1].id == 10
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_list_field_ids': [3, 4]},
             {'id': 2, 'some_chain_list_field_ids': [4, 5]}],
            [{'id': 3, 'some_related_field_id': [6, ''], 'some_chain_field_id': [7, '']},
             {'id': 4, 'some_related_field_id': [8, ''], 'some_chain_field_id': [9, '']},
             {'id': 5, 'some_related_field_id': [10, ''], 'some_chain_field_id': [11, '']}],
            [{'id': 7}, {'id': 9}, {'id': 11}],
            [{'id': 6}, {'id': 8}, {'id': 10}]))
    def test_multiple_prefetches_same_model_4(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_list_field__some_related_field',
                                   'some_chain_list_field__some_chain_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4, 5], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [7, 9, 11], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [6, 8, 10], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_list_field[0].id == 3
        assert instances[0].some_chain_list_field[1].id == 4
        assert instances[0].some_chain_list_field[0].some_related_field.id == 6
        assert instances[0].some_chain_list_field[0].some_chain_field.id == 7
        assert instances[0].some_chain_list_field[1].some_related_field.id == 8
        assert instances[0].some_chain_list_field[1].some_chain_field.id == 9
        assert instances[1].some_chain_list_field[0].id == 4
        assert instances[1].some_chain_list_field[1].id == 5
        assert instances[1].some_chain_list_field[0].some_related_field.id == 8
        assert instances[1].some_chain_list_field[0].some_chain_field.id == 9
        assert instances[1].some_chain_list_field[1].some_related_field.id == 10
        assert instances[1].some_chain_list_field[1].some_chain_field.id == 11
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_list_field_ids': [3, 4]},
             {'id': 2, 'some_chain_list_field_ids': [4, 5]}],
            [{'id': 3, 'some_related_field_id': [6, ''], 'some_list_field_ids': [9, 10]},
             {'id': 4, 'some_related_field_id': [7, ''], 'some_list_field_ids': [11, 12]},
             {'id': 5, 'some_related_field_id': [8, ''], 'some_list_field_ids': [13, 14]}],
            [{'id': 9}, {'id': 10}, {'id': 11}, {'id': 12}, {'id': 13}, {'id': 14}],
            [{'id': 6}, {'id': 7}, {'id': 8}]))
    def test_multiple_prefetches_same_model_5(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_list_field__some_related_field',
                                   'some_chain_list_field__some_list_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4, 5], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [9, 10, 11, 12, 13, 14], fields=list(ModelBase.all_fields_odoo_names())),
            call('model.base', 'read', [6, 7, 8], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_list_field[0].id == 3
        assert instances[0].some_chain_list_field[1].id == 4
        assert instances[0].some_chain_list_field[0].some_related_field.id == 6
        assert instances[0].some_chain_list_field[0].some_list_field[0].id == 9
        assert instances[0].some_chain_list_field[0].some_list_field[1].id == 10
        assert instances[0].some_chain_list_field[1].some_related_field.id == 7
        assert instances[0].some_chain_list_field[1].some_list_field[0].id == 11
        assert instances[0].some_chain_list_field[1].some_list_field[1].id == 12
        assert instances[1].some_chain_list_field[0].id == 4
        assert instances[1].some_chain_list_field[1].id == 5
        assert instances[1].some_chain_list_field[0].some_related_field.id == 7
        assert instances[1].some_chain_list_field[0].some_list_field[0].id == 11
        assert instances[1].some_chain_list_field[0].some_list_field[1].id == 12
        assert instances[1].some_chain_list_field[1].some_related_field.id == 8
        assert instances[1].some_chain_list_field[1].some_list_field[0].id == 13
        assert instances[1].some_chain_list_field[1].some_list_field[1].id == 14
        spy_execute.assert_not_called()

    @pytest.mark.connection_returns(
        *SomeModel.populate_odoo_return_values(
            [{'id': 1, 'some_chain_field_id': [3, '']},
             {'id': 2, 'some_chain_field_id': [4, '']}],
            [{'id': 3, 'some_chain_field_id': [5, '']},
             {'id': 4, 'some_chain_field_id': [6, '']}],
            [{'id': 5, 'some_related_field_id': [7, ''], 'some_chain_field_id': [9, '']},
             {'id': 6, 'some_related_field_id': [8, ''], 'some_chain_field_id': [10, '']}],
            [{'id': 9}, {'id': 10}],
            [{'id': 7}, {'id': 8}]))
    def test_multiple_prefetches_same_model_6(self, spy_execute: MagicMock):
        instances = list(QuerySet(SomeModel)
                         .prefetch('some_chain_field__some_chain_field__some_related_field',
                                   'some_chain_field__some_chain_field__some_chain_field'))
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [3, 4], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [5, 6], fields=list(SomeModel.all_fields_odoo_names())),
            call('some.model', 'read', [9, 10], fields=list(SomeModel.all_fields_odoo_names())),
            call('model.base', 'read', [7, 8], fields=list(ModelBase.all_fields_odoo_names())),
        ]
        spy_execute.reset_mock()
        assert instances[0].some_chain_field.id == 3
        assert instances[0].some_chain_field.some_chain_field.id == 5
        assert instances[0].some_chain_field.some_chain_field.some_related_field.id == 7
        assert instances[0].some_chain_field.some_chain_field.some_chain_field.id == 9
        assert instances[1].some_chain_field.id == 4
        assert instances[1].some_chain_field.some_chain_field.id == 6
        assert instances[1].some_chain_field.some_chain_field.some_related_field.id == 8
        assert instances[1].some_chain_field.some_chain_field.some_chain_field.id == 10
        spy_execute.assert_not_called()

    def test_equality(self):
        assert QuerySet(SomeModel) == QuerySet(SomeModel)
        assert QuerySet(SomeModel) != [SomeModel(id=1)]
        assert (QuerySet(SomeModel).limit(3).filter(id=3, name='tut')
                == QuerySet(SomeModel).filter(id=3).filter(name='tut').limit(3))
        assert QuerySet(SomeModel).filter(id=3).filter(name='tut') != QuerySet(SomeModel).filter(name='tut', id=3)

    @pytest.mark.connection_returns([{'id': 1}, {'id': 2}, {'id': 3}],
                                    True)
    def test_delete(self, spy_execute: MagicMock):
        QuerySet(SomeModel).delete()
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [], fields=['id']),
            call('some.model', 'unlink', [1, 2, 3])
        ]

    @pytest.mark.connection_returns([{'id': 1}, {'id': 2}],
                                    True)
    def test_delete_respects_filters(self, spy_execute: MagicMock):
        QuerySet(SomeModel).filter(some_field='pouet').delete()
        assert spy_execute.call_args_list == [
            call('some.model', 'search_read', [('some_field', '=', 'pouet')], fields=['id']),
            call('some.model', 'unlink', [1, 2])
        ]


class TestManager:

    def test_all(self):
        assert SomeModel.objects.all() == QuerySet(SomeModel)

    def test_filter(self):
        assert SomeModel.objects.filter(name='tut') == QuerySet(SomeModel).filter(name='tut')

    def test_values(self):
        assert SomeModel.objects.values('some_field') == QuerySet(SomeModel).values('some_field')

    def test_limit(self):
        assert SomeModel.objects.limit(3) == QuerySet(SomeModel).limit(3)

    def test_order_by(self):
        assert SomeModel.objects.order_by('some_field') == QuerySet(SomeModel).order_by('some_field')

    @pytest.mark.connection_returns([{'id': 1}])
    def test_get(self, spy_execute: MagicMock):
        assert SomeModel.objects.values().get(id=1) == SomeModel(id=1)
        spy_execute.assert_called_once_with('some.model', 'read', [1], fields=[])


class TestModelBase:

    def test_all_fields_odoo_names(self):
        assert list(SomeModel.all_fields_odoo_names()) == ['id', 'some_field', 'some_related_field_id',
                                                           'some_list_field_ids', 'named_string', 'some_null_field',
                                                           'some_b64_field', 'some_boolean_field', 'some_decimal_field',
                                                           'some_date_field', 'some_chain_field_id',
                                                           'some_chain_list_field_ids',
                                                           'some_nullable_related_field_id', 'some_datetime_field',
                                                           'some_resolvable_field_id']

    def test_field_odoo_names(self):
        assert list(SomeModel.field_odoo_names('some_field')) == ['some_field']
        assert list(SomeModel.field_odoo_names('some_named_field')) == ['named_string']

    def test_model_constructor_set_attributes(self, basic_instance: SomeModel):
        assert basic_instance.id == 1
        assert basic_instance.some_field == 'tut'
        assert basic_instance.some_related_field_id == 2
        assert basic_instance.some_list_field_ids == [3, 4]
        assert basic_instance.some_named_field == 'pouet'

    def test_model_str(self, basic_instance: SomeModel):
        assert str(basic_instance) == 'SomeModel(1)'

    def test_model_pprint1(self, basic_instance: SomeModel):
        expected = """\
SomeModel {
  id: 1
  some_field: 'tut'
  some_related_field: ModelBase(2)
  some_list_field: ModelBase[3, 4]
  some_named_field: 'pouet'
}\
"""
        assert basic_instance.pprint() == expected

    def test_model_pprint2(self):
        instance = SomeModel(id=1, some_list_field=[])
        expected = """\
SomeModel {
  id: 1
  some_list_field: ModelBase[]
}\
"""
        assert instance.pprint() == expected

    def test_model_pprint3(self):
        instance = SomeModel()
        assert instance.pprint() == 'SomeModel {}'

    def test_equality(self):
        instance1 = ModelBase(id=1)
        instance1b = ModelBase(id=1)
        instance2 = ModelBase(id=2)
        assert instance1 == instance1b
        assert instance1 != instance2
        assert instance1b != instance2
        assert instance1 != Model(id=1)

    def test_save(self, spy_execute: MagicMock, basic_instance: SomeModel):
        basic_instance.some_field = 'tutturu'
        basic_instance.some_related_field = ModelBase(id=3)
        basic_instance.some_list_field = [ModelBase(id=5), ModelBase(id=6)]
        basic_instance.some_named_field = 'ohayou~'
        basic_instance.save()

        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_field': 'tutturu',
            'some_related_field_id': 3,
            'some_list_field_ids': [(6, '_', [5, 6])],
            'named_string': 'ohayou~',
        })

    def test_save_only_what_changed(self, spy_execute: MagicMock, basic_instance: SomeModel):
        basic_instance.some_field = 'tutturu'
        basic_instance.save()

        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_field': 'tutturu',
        })

    def test_save_based_on_raw_values(self, spy_execute: MagicMock, basic_instance: SomeModel):
        basic_instance.some_related_field_id = 3
        basic_instance.some_list_field_ids = [5, 6]
        basic_instance.save()

        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_related_field_id': 3,
            'some_list_field_ids': [(6, '_', [5, 6])],
        })

    def test_cannot_save_id(self, basic_instance: SomeModel):
        basic_instance.id = 2
        with pytest.raises(InvalidModelState):
            basic_instance.save()

    @pytest.mark.connection_returns(1)
    def test_create_instance(self, spy_execute: MagicMock):
        basic_instance = SomeModel(some_field='tut')
        basic_instance.save()

        spy_execute.assert_called_once_with('some.model', 'create', {
            'some_field': 'tut',
        })
        spy_execute.reset_mock()
        assert basic_instance.id == 1
        basic_instance.save()
        spy_execute.assert_not_called()

    def test_save_reset_initial_values(self, spy_execute: MagicMock, basic_instance: SomeModel):
        initial_value = basic_instance.some_field

        basic_instance.some_field = 'tutturu'
        basic_instance.save()

        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_field': 'tutturu',
        })
        spy_execute.reset_mock()

        basic_instance.save()

        spy_execute.assert_not_called()

        basic_instance.some_field = initial_value
        basic_instance.save()

        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_field': initial_value,
        })

    def test_form_odoo_(self, basic_instance: SomeModel):
        basic_instance2 = SomeModel(id=1, some_field='tut', some_related_field=ModelBase(id=2),
                                    some_list_field=[ModelBase(id=3), ModelBase(id=4)], some_named_field='pouet')
        assert basic_instance == basic_instance2

    def test_save_turns_none_into_false(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(id=1, some_null_field='Pouet')
        assert instance.some_null_field is not None
        instance.some_null_field = None
        instance.save()
        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_null_field': False,
        })

    def test_save_turns_none_into_false_even_for_related_fields(self, spy_execute: MagicMock):
        instance = SomeModel.from_odoo(id=1, some_nullable_related_field_id=[2, 'tut'])
        assert instance.some_nullable_related_field_id is not None
        instance.some_nullable_related_field = None
        instance.save()
        spy_execute.assert_called_once_with('some.model', 'write', 1, {
            'some_nullable_related_field_id': False,
        })

    def test_delete(self, spy_execute: MagicMock, basic_instance: SomeModel):
        basic_instance.delete()
        spy_execute.assert_called_once_with('some.model', 'unlink', [1])


class TestModel:

    @pytest.mark.connection_returns([{'id': 2, 'name': '', 'datas': ''}])
    def test_attachments(self, spy_execute: MagicMock):
        instance = Model(id=1)
        assert list(instance.attachments) == [Attachment(id=2, name='', content=b'')]
        spy_execute.assert_called_once_with('ir.attachment', 'search_read',
                                            [('res_model', '=', 'model'), ('res_id', '=', 1)],
                                            fields=['id', 'name', 'datas'])

    @pytest.mark.connection_returns({'result': b64encode(b'tutturu')})
    def test_render_report(self, spy_render_report: MagicMock):
        assert Model(id=1).render_report('a_', lang='gura') == b'tutturu'

        spy_render_report.assert_called_once_with('a_', 1, lang='gura')

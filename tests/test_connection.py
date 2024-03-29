import pytest

from odoo_orm.connection import OdooConnection
from odoo_orm.errors import (
    OdooConnectionAlreadyExists, OdooConnectionError, OdooConnectionNotConnected, UnsafeOperationNotAllowed,
)


class TestConnectionCreation:

    @pytest.fixture(autouse=True)
    def reset_connection(self):
        OdooConnection.CONNECTION = None

    def test_create_one_connection(self):
        odoo = OdooConnection()
        assert odoo is not None

    def test_create_two_connections(self):
        OdooConnection()
        with pytest.raises(OdooConnectionAlreadyExists):
            OdooConnection()

    def test_create_with_get_connection(self):
        odoo = OdooConnection.get_connection()
        assert odoo is not None

    def test_create_and_get_back(self):
        odoo = OdooConnection()
        odoo2 = OdooConnection.get_connection()
        assert odoo == odoo2

    def test_get_connection_twice(self):
        odoo = OdooConnection.get_connection()
        odoo2 = OdooConnection.get_connection()
        assert odoo == odoo2


class TestConnectionConnection:

    def test_do_request_but_not_connected(self, odoo: OdooConnection):
        with pytest.raises(OdooConnectionNotConnected):
            odoo.execute('', '')

    def test_do_render_report_but_not_connected(self, odoo: OdooConnection):
        with pytest.raises(OdooConnectionNotConnected):
            odoo.render_report('', 0)

    def test_connect(self, odoo: OdooConnection):
        odoo.connect('', '', '', '')
        assert odoo.uid == 1

    def test_connect_wrong_login_password(self, auth_forbidden_odoo: OdooConnection):
        with pytest.raises(OdooConnectionError):
            auth_forbidden_odoo.connect('', '', '', '')

    def test_connect_twice(self, odoo: OdooConnection):
        odoo.connect('', '', '', '')
        with pytest.raises(OdooConnectionError):
            odoo.connect('', '', '', '')


class TestConnectionUsage:

    def test_request_must_pass_arguments(self, connected_odoo: OdooConnection):
        args = name, action, params, options = 'fixture.model', 'search', ([['id', '=', 1200]],), {'fields': 'a_'}
        return_value = connected_odoo.execute(name, action, *params, **options)
        assert return_value == ('', 1, '', *args)

    def test_render_report_must_pass_arguments(self, connected_safe_odoo: OdooConnection):
        args = report_name, [model_id], options = 'fixture.render_model', [7], {'tutturu': 'ohayou~'}
        return_value = connected_safe_odoo.render_report(report_name, model_id, **options)
        assert return_value == ('', 1, '', *args)


class TestConnectionSafety:

    def test_unsafe_odoo_can_only_read_and_search(self, connected_odoo: OdooConnection):
        for action in ('search', 'search_read', 'read'):
            connected_odoo.execute('some.model', action)

        for action in ('create', 'write', 'unlink'):
            with pytest.raises(UnsafeOperationNotAllowed):
                connected_odoo.execute('some.model', action)

        with pytest.raises(UnsafeOperationNotAllowed):
            connected_odoo.render_report('some.model', 1)

    def test_connected_safe_odoo_can_do_anything(self, connected_safe_odoo: OdooConnection):
        for action in ('search', 'search_read', 'read', 'create', 'write', 'unlink'):
            connected_safe_odoo.execute('some.model', action)

        connected_safe_odoo.render_report('some.model', 1)

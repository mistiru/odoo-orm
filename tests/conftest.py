import pytest
from pytest_mock import MockerFixture

from odoo_orm.connection import OdooConnection


class OdooServerProxy:

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def authenticate(self, *args, **kwargs):
        return 1

    def execute_kw(self, db, uid, password, model, action, params, options):
        return db, uid, password, model, action, params, options

    def render_report(self, db, uid, password, report_name, model_ids, options):
        return db, uid, password, report_name, model_ids, options


@pytest.fixture
def odoo(mocker: MockerFixture):
    connection = OdooConnection.get_connection()
    mocker.patch('odoo_orm.connection.ServerProxy', OdooServerProxy)
    yield connection
    OdooConnection.CONNECTION = None


@pytest.fixture
def auth_forbidden_odoo(odoo: OdooConnection, mocker: MockerFixture):
    mocker.patch('odoo_orm.connection.ServerProxy.authenticate', lambda *args, **kwargs: None)
    return odoo


@pytest.fixture
def connected_odoo(odoo: OdooConnection):
    odoo.connect('', '', '', '')
    return odoo


@pytest.fixture
def connected_safe_odoo(odoo: OdooConnection):
    odoo.connect('', '', '', '', True)
    return odoo


class OdooConnectionProxy:
    def __init__(self, data):
        self.data = data

    def _next_data(self):
        if self.data:
            data = next(self.data)
            if isinstance(data, BaseException):
                raise data
            else:
                return data
        else:
            return None

    def execute(self, *args, **kwargs):
        return self._next_data()

    def render_report(self, *args, **kwargs):
        return self._next_data()


@pytest.fixture
def odoo_connection_proxy(request, mocker):
    marker = request.node.get_closest_marker('connection_returns')
    data = marker and iter(marker.args)

    proxy = OdooConnectionProxy(data)
    mocker.patch('odoo_orm.orm.connection', proxy)
    return proxy


@pytest.fixture
def spy_execute(odoo_connection_proxy, mocker):
    return mocker.spy(odoo_connection_proxy, 'execute')


@pytest.fixture
def spy_render_report(odoo_connection_proxy, mocker):
    return mocker.spy(odoo_connection_proxy, 'render_report')

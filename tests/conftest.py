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
    yield odoo


@pytest.fixture
def connected_odoo(odoo: OdooConnection):
    odoo.connect('', '', '', '')
    yield odoo

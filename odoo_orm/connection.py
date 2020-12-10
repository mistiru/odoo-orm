import logging
from typing import Any
from xmlrpc.client import ServerProxy

from odoo_orm.errors import OdooConnectionError

logger = logging.getLogger(__name__)


class OdooConnection:
    CONNECTION: 'OdooConnection' = None

    db: str
    uid: int
    password: str
    models: ServerProxy
    reports: ServerProxy

    def __init__(self) -> None:
        if OdooConnection.CONNECTION:
            raise Exception('Connection already set up!')

        OdooConnection.CONNECTION = self

    @classmethod
    def get_connection(cls) -> 'OdooConnection':
        if OdooConnection.CONNECTION:
            return OdooConnection.CONNECTION
        else:
            return cls()

    def connect(self, url: str, db: str, user: str, password: str) -> None:
        with ServerProxy(f'{url}/xmlrpc/2/common') as common:
            uid = common.authenticate(db, user, password, {})
            if uid:
                self.db = db
                self.uid = uid
                self.password = password
                self.models = ServerProxy(f'{url}/xmlrpc/2/object')
                self.reports = ServerProxy(f'{url}/xmlrpc/2/report')
            else:
                raise OdooConnectionError('Wrong email or password')

    def execute(self, model: str, action: str, *params, **options) -> list[dict[str, Any]]:
        with self.models:
            logger.debug(f'models.execute_kw({self.db!r}, {self.uid!r}, {self.password!r}, {model!r}, {action!r},'
                         f' {params!r}, {options!r})')
            return self.models.execute_kw(self.db, self.uid, self.password, model, action, params, options)

    def render_report(self, report_name: str, model_id: int, **options) -> dict[str, Any]:
        with self.reports:
            logger.debug(f'reports.render_report({self.db!r}, {self.uid!r}, {self.password!r}, {report_name!r},'
                         f' {[model_id]!r}, {options!r})')
            return self.reports.render_report(self.db, self.uid, self.password, report_name, [model_id], options)

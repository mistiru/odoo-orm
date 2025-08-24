import logging
from typing import Any
from xmlrpc.client import ServerProxy

from odoo_orm.errors import (
    OdooConnectionAlreadyExists, OdooConnectionError, OdooConnectionNotConnected, UnsafeOperationNotAllowed,
)

logger = logging.getLogger(__name__)


class OdooConnection:
    CONNECTION: 'OdooConnection' = None

    db: str
    uid: int
    url: str
    password: str
    safe: bool

    def __init__(self) -> None:
        if OdooConnection.CONNECTION:
            raise OdooConnectionAlreadyExists('Connection already set up!')

        OdooConnection.CONNECTION = self

    @classmethod
    def get_connection(cls) -> 'OdooConnection':
        if OdooConnection.CONNECTION:
            return OdooConnection.CONNECTION
        else:
            return cls()

    def connect(self, url: str, db: str, user: str, password: str, safe: bool = False) -> None:
        if hasattr(self, 'uid'):
            raise OdooConnectionError('Already connected')

        with ServerProxy(f'{url}/xmlrpc/2/common') as common:
            uid = common.authenticate(db, user, password, {})
            if uid:
                self.db = db
                self.uid = uid
                self.url = url
                self.password = password
                self.safe = safe
            else:
                raise OdooConnectionError('Wrong email or password')

    @property
    def models(self):
        return ServerProxy(f'{self.url}/xmlrpc/2/object')

    @property
    def reports(self):
        return ServerProxy(f'{self.url}/xmlrpc/2/report')

    def execute(self, model: str, action: str, *params, **options) -> list[dict[str, Any]]:
        if not hasattr(self, 'models'):
            raise OdooConnectionNotConnected('You must connect before doing any request')

        if not self.safe and action not in ('search', 'search_read', 'read', 'read_group'):
            raise UnsafeOperationNotAllowed('Trying to perform an unsafe operation in unsafe environment')

        with self.models as proxy:
            logger.debug(f'models.execute_kw({self.db!r}, {self.uid!r}, {self.password!r}, {model!r}, {action!r},'
                         f' {params!r}, {options!r})')
            return proxy.execute_kw(self.db, self.uid, self.password, model, action, params, options)

    def render_report(self, report_name: str, model_id: int, **options) -> dict[str, Any]:
        if not hasattr(self, 'reports'):
            raise OdooConnectionNotConnected('You must connect before doing any request')

        if not self.safe:
            raise UnsafeOperationNotAllowed('Trying to perform an unsafe operation in unsafe environment')

        with self.reports as proxy:
            logger.debug(f'reports.render_report({self.db!r}, {self.uid!r}, {self.password!r}, {report_name!r},'
                         f' {[model_id]!r}, {options!r})')
            return proxy.render_report(self.db, self.uid, self.password, report_name, [model_id], options)

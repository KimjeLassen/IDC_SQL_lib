from sqlalchemy.orm import Session
from app.database.models import UserRolesMapping, SystemRoleAssignments, SystemRoles
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def fetch_data(db: Session):
    try:
        # Construct the ORM query
        query = (
            db.query(
                UserRolesMapping.user_id.label("user_id"),
                SystemRoles.name.label("system_role_name"),
            )
            .join(
                SystemRoleAssignments,
                UserRolesMapping.user_role_id == SystemRoleAssignments.user_role_id,
            )
            .join(SystemRoles, SystemRoleAssignments.system_role_id == SystemRoles.id)
        )

        # Execute the query and load results into a DataFrame
        df = pd.read_sql(query.statement, db.bind)
        return df
    except Exception:
        logger.error("An error occurred while fetching data:", exc_info=True)
        return None

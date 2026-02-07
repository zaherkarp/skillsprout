"""Enhanced model registry with versioning, promotion, and rollback.

This module provides a ``ModelRegistry`` service layer on top of the
``model_registry`` database table.  Key capabilities:

- Store artifacts with rich metadata (version, trained_at, data_range,
  feature_set, eval_metrics, status).
- Atomic promotion: candidate -> production with auto-demotion of the
  previous production model.
- Feature schema versioning so training runs are reproducible.
- Rollback to any previously promoted model version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SyncSessionLocal
from app.models.models import ModelRegistry as ModelRegistryRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class ModelStatus(str, Enum):
    """Lifecycle status of a registered model."""

    CANDIDATE = "candidate"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


@dataclass
class FeatureSchema:
    """Describes the feature contract a model was trained on.

    Storing the exact feature list, their types, and any transformations
    ensures that a model can be reproduced or validated at serving time.
    """

    version: str
    feature_names: List[str]
    feature_types: Dict[str, str] = field(default_factory=dict)
    transformations: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "version": self.version,
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "transformations": self.transformations,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureSchema:
        """Deserialise from a dictionary."""
        return cls(
            version=data.get("version", "unknown"),
            feature_names=data.get("feature_names", []),
            feature_types=data.get("feature_types", {}),
            transformations=data.get("transformations", {}),
            description=data.get("description", ""),
        )


@dataclass
class ModelArtifact:
    """Rich descriptor for a registered model artifact."""

    model_version: str
    trained_at: datetime
    artifact_path: str
    status: ModelStatus
    training_samples: int
    eval_metrics: Dict[str, float]
    feature_schema: Optional[FeatureSchema] = None
    data_range_start: Optional[datetime] = None
    data_range_end: Optional[datetime] = None
    notes: str = ""
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary (stored in metrics_json)."""
        return {
            "status": self.status.value,
            "eval_metrics": self.eval_metrics,
            "feature_schema": (
                self.feature_schema.to_dict() if self.feature_schema else None
            ),
            "data_range_start": (
                self.data_range_start.isoformat() if self.data_range_start else None
            ),
            "data_range_end": (
                self.data_range_end.isoformat() if self.data_range_end else None
            ),
            "promoted_at": (
                self.promoted_at.isoformat() if self.promoted_at else None
            ),
            "demoted_at": (
                self.demoted_at.isoformat() if self.demoted_at else None
            ),
        }


# ---------------------------------------------------------------------------
# Registry service
# ---------------------------------------------------------------------------

class ModelRegistryService:
    """Service layer for model lifecycle management.

    All methods that touch the database accept an optional ``db`` session.
    When omitted a fresh sync session is created and closed automatically.

    Example::

        registry = ModelRegistryService()
        registry.register(artifact)
        registry.promote("v2.4_candidate")
        registry.rollback("v2.3")
    """

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        artifact: ModelArtifact,
        db: Optional[Session] = None,
    ) -> ModelRegistryRow:
        """Register a new model artifact.

        The model is stored with status ``CANDIDATE`` by default.  Use
        :meth:`promote` to move it to ``PRODUCTION``.

        Args:
            artifact: Descriptor for the model to register.
            db: Optional SQLAlchemy session.

        Returns:
            The newly created ``ModelRegistryRow``.

        Raises:
            ValueError: If a model with the same version already exists.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            existing = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.model_version == artifact.model_version)
                .first()
            )
            if existing is not None:
                raise ValueError(
                    f"Model version '{artifact.model_version}' is already "
                    "registered.  Use a different version string."
                )

            metrics_json = artifact.eval_metrics.copy()
            metrics_json.update(artifact.to_dict())

            row = ModelRegistryRow(
                model_version=artifact.model_version,
                trained_at=artifact.trained_at,
                training_samples=artifact.training_samples,
                metrics_json=metrics_json,
                artifact_path=artifact.artifact_path,
                is_active=(artifact.status == ModelStatus.PRODUCTION),
                notes=artifact.notes,
            )
            db.add(row)
            db.commit()
            db.refresh(row)

            logger.info(
                "Registered model %s (status=%s, samples=%d)",
                artifact.model_version,
                artifact.status.value,
                artifact.training_samples,
            )
            return row

        except Exception:
            db.rollback()
            raise
        finally:
            if close_db:
                db.close()

    # ------------------------------------------------------------------
    # Promotion / demotion
    # ------------------------------------------------------------------

    def promote(
        self,
        model_version: str,
        db: Optional[Session] = None,
    ) -> ModelRegistryRow:
        """Atomically promote a candidate model to production.

        This demotes the currently active production model (if any) and
        marks the specified version as active.

        Args:
            model_version: Version string of the model to promote.
            db: Optional SQLAlchemy session.

        Returns:
            The promoted ``ModelRegistryRow``.

        Raises:
            ValueError: If the model version does not exist or is already
                in production.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            candidate = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.model_version == model_version)
                .first()
            )
            if candidate is None:
                raise ValueError(f"Model version '{model_version}' not found")
            if candidate.is_active:
                raise ValueError(
                    f"Model version '{model_version}' is already in production"
                )

            # Demote all currently active models.
            now = datetime.utcnow()
            active_models = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.is_active.is_(True))
                .all()
            )
            for active in active_models:
                active.is_active = False
                meta = active.metrics_json or {}
                meta["status"] = ModelStatus.ARCHIVED.value
                meta["demoted_at"] = now.isoformat()
                active.metrics_json = meta
                logger.info("Demoted model %s", active.model_version)

            # Promote candidate.
            candidate.is_active = True
            meta = candidate.metrics_json or {}
            meta["status"] = ModelStatus.PRODUCTION.value
            meta["promoted_at"] = now.isoformat()
            candidate.metrics_json = meta

            db.commit()
            db.refresh(candidate)

            logger.info("Promoted model %s to production", model_version)
            return candidate

        except Exception:
            db.rollback()
            raise
        finally:
            if close_db:
                db.close()

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(
        self,
        target_version: str,
        db: Optional[Session] = None,
    ) -> ModelRegistryRow:
        """Rollback production to a previously promoted model.

        The current production model is demoted and the target version is
        re-promoted.

        Args:
            target_version: Version string to roll back to.
            db: Optional SQLAlchemy session.

        Returns:
            The re-promoted ``ModelRegistryRow``.

        Raises:
            ValueError: If the target version does not exist.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            target = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.model_version == target_version)
                .first()
            )
            if target is None:
                raise ValueError(
                    f"Target rollback version '{target_version}' not found"
                )

            # Demote current production model(s).
            now = datetime.utcnow()
            active_models = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.is_active.is_(True))
                .all()
            )
            for active in active_models:
                active.is_active = False
                meta = active.metrics_json or {}
                meta["status"] = ModelStatus.ROLLED_BACK.value
                meta["demoted_at"] = now.isoformat()
                active.metrics_json = meta
                logger.info(
                    "Rolled back model %s (was production)", active.model_version
                )

            # Re-activate target.
            target.is_active = True
            meta = target.metrics_json or {}
            meta["status"] = ModelStatus.PRODUCTION.value
            meta["promoted_at"] = now.isoformat()
            target.metrics_json = meta

            db.commit()
            db.refresh(target)

            logger.info("Rolled back to model %s", target_version)
            return target

        except Exception:
            db.rollback()
            raise
        finally:
            if close_db:
                db.close()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_model(
        self, db: Optional[Session] = None
    ) -> Optional[ModelRegistryRow]:
        """Return the currently active (production) model, or ``None``.

        Args:
            db: Optional SQLAlchemy session.

        Returns:
            The active ``ModelRegistryRow`` or ``None``.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            return (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.is_active.is_(True))
                .order_by(ModelRegistryRow.trained_at.desc())
                .first()
            )
        finally:
            if close_db:
                db.close()

    def get_model(
        self,
        model_version: str,
        db: Optional[Session] = None,
    ) -> Optional[ModelRegistryRow]:
        """Look up a single model by version.

        Args:
            model_version: Version string.
            db: Optional SQLAlchemy session.

        Returns:
            The ``ModelRegistryRow`` or ``None``.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            return (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.model_version == model_version)
                .first()
            )
        finally:
            if close_db:
                db.close()

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        limit: int = 50,
        db: Optional[Session] = None,
    ) -> List[ModelRegistryRow]:
        """List registered models with optional status filter.

        Args:
            status: Filter by lifecycle status stored in ``metrics_json``.
            limit: Maximum number of rows to return.
            db: Optional SQLAlchemy session.

        Returns:
            List of ``ModelRegistryRow`` objects, newest first.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            query = db.query(ModelRegistryRow).order_by(
                ModelRegistryRow.trained_at.desc()
            )

            if status is not None:
                if status == ModelStatus.PRODUCTION:
                    query = query.filter(ModelRegistryRow.is_active.is_(True))
                else:
                    query = query.filter(ModelRegistryRow.is_active.is_(False))

            return query.limit(limit).all()
        finally:
            if close_db:
                db.close()

    # ------------------------------------------------------------------
    # Feature schema versioning
    # ------------------------------------------------------------------

    def set_feature_schema(
        self,
        model_version: str,
        schema: FeatureSchema,
        db: Optional[Session] = None,
    ) -> ModelRegistryRow:
        """Attach or update the feature schema for a registered model.

        Args:
            model_version: Version string.
            schema: The ``FeatureSchema`` to store.
            db: Optional SQLAlchemy session.

        Returns:
            The updated ``ModelRegistryRow``.

        Raises:
            ValueError: If the model version does not exist.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            row = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.model_version == model_version)
                .first()
            )
            if row is None:
                raise ValueError(f"Model version '{model_version}' not found")

            meta = row.metrics_json or {}
            meta["feature_schema"] = schema.to_dict()
            row.metrics_json = meta

            db.commit()
            db.refresh(row)

            logger.info(
                "Feature schema v%s attached to model %s",
                schema.version,
                model_version,
            )
            return row

        except Exception:
            db.rollback()
            raise
        finally:
            if close_db:
                db.close()

    def get_feature_schema(
        self,
        model_version: str,
        db: Optional[Session] = None,
    ) -> Optional[FeatureSchema]:
        """Retrieve the feature schema for a registered model.

        Args:
            model_version: Version string.
            db: Optional SQLAlchemy session.

        Returns:
            The ``FeatureSchema`` or ``None`` if not set.
        """
        close_db = db is None
        db = db or SyncSessionLocal()

        try:
            row = (
                db.query(ModelRegistryRow)
                .filter(ModelRegistryRow.model_version == model_version)
                .first()
            )
            if row is None:
                return None
            meta = row.metrics_json or {}
            schema_dict = meta.get("feature_schema")
            if schema_dict is None:
                return None
            return FeatureSchema.from_dict(schema_dict)
        finally:
            if close_db:
                db.close()

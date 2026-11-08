"""Dynamic prompt management: domains, detection, resolution, seeding."""

from app.services.prompt_management.detector import detect_domain
from app.services.prompt_management.resolver import ResolvedPrompts, resolve_prompts

__all__ = ["detect_domain", "resolve_prompts", "ResolvedPrompts"]

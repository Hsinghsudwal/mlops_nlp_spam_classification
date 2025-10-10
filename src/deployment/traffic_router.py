import random
import yaml
from utils.logger import logger


class TrafficRouter:
    """Manages traffic routing for different deployment strategies."""

    def __init__(self, config):
        self.config = config
        self.routing_cfg = self.config.get("policy.traffic_routing")
        self.shadow_enabled = self.routing_cfg.get("shadow_enabled", False)
        self.available_targets = self.routing_cfg.get(
            "targets", ["staging", "production"]
        )
        self.canary_ratio = self.routing_cfg.get("canary_ratio", 0.1)

    def route(self, strategy: str) -> str:
        """Determine routing target based on deployment strategy."""
        logger.info(f"Routing traffic with strategy: {strategy}")

        if strategy == "direct":
            target = "production"
        elif strategy == "canary":
            target = "staging" if random.random() < self.canary_ratio else "production"
        elif strategy == "shadow":
            # Shadow deployment: main traffic to production, shadow to staging
            target = "production"
        else:
            # Fallback to production
            target = "production"

        logger.info(f"Traffic routed to: {target}")
        return target

    def should_shadow(self, strategy: str = "shadow") -> bool:
        """Determine if shadow mode should be enabled."""
        return strategy == "shadow" and self.shadow_enabled

    def get_routing_config(self, strategy: str):
        """Get complete routing configuration."""
        return {
            "strategy": strategy,
            "target": self.route(strategy),
            "shadow_enabled": self.should_shadow(strategy),
            "canary_ratio": self.canary_ratio if strategy == "canary" else None,
            "available_targets": self.available_targets,
        }

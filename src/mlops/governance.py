import yaml


class Governance:
    """Simple governance engine for ML policies."""

    def __init__(self, config):
        self.policies = config

        self.evaluate_policies = self.policies.get("policy.evaluate", {})
        self.deployment_policies = self.policies.get("deployment", {})
        self.allowed_strategies = set(
            self.deployment_policies.get(
                "allowed_strategies", ["canary", "shadow", "direct"]
            )
        )
        self.default_strategy = self.deployment_policies.get(
            "default_strategy", "direct"
        )
        # self.monitoring_cfg = self.config.get("monitoring", {})

    def decision(self):
        return {"approved": True, "reasons": [], "actions": []}

    def policy_evaluate(self, metrics):
        decision = self.decision()

        validate_threshold = self.evaluate_policies.get("validate_threshold", 0.8)
        accuracy = metrics.get("accuracy", 0.80)

        if accuracy < validate_threshold:
            decision["approved"] = False
            decision["reasons"].append(
                f"Accuracy {accuracy} is below minimum {validate_threshold}"
            )
            decision["actions"].append("Improve training/ hypertuning")

        return decision

    def policy_deployment(self, strategy):
        decision = self.decision()

        if strategy not in self.allowed_strategies:
            decision["approved"] = False
            decision["reasons"].append(
                f"Deployment strategy '{strategy}' not in allowed strategies: {list(self.allowed_strategies)}"
            )
            decision["actions"].append(
                "Use an allowed strategy or update governance policy"
            )

        return decision
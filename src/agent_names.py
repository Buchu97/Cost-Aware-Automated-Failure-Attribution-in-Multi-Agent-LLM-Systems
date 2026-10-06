"""Conversation actor identities, distinct from hand-crafted message roles."""


def canonical_agent(name, trace: dict):
    # The benchmark labels the actor Orchestrator; its role suffix describes
    # thought, recipient or termination messages rather than a different agent.
    if trace['is_handcrafted'] and isinstance(name, str) and name.startswith('Orchestrator ('):
        return 'Orchestrator'
    return name


def allowed_agents(trace: dict) -> set[str]:
    key = 'role' if trace['is_handcrafted'] else 'name'
    return {canonical_agent(m[key], trace) for m in trace['messages'] if isinstance(m.get(key), str)}

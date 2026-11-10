"""Optional upstream τ² stateful environment bridge; no simulated tool results."""
from copy import deepcopy


class Tau2EnvironmentBridge:
    """Keep private task criteria separate from public policy/tools/messages.

    Requires the separately installed, pinned τ² runtime. This is a sequential
    environment interface, not a speech user simulator or full orchestrator.
    """

    def __init__(self, environment_constructor, task):
        self._constructor = environment_constructor
        self._task = task.model_copy(deep=True)
        initial = self._task.initial_state
        self._initialization_data = initial.initialization_data if initial else None
        self._initialization_actions = initial.initialization_actions if initial else None
        self._history = deepcopy((initial.message_history or []) if initial else [])
        self._environment = self._new_environment(self._history)
        self._call_ids = {
            call.id for message in self._history
            for call in (getattr(message, 'tool_calls', None) or [])
        }

    def _new_environment(self, history):
        environment = self._constructor()
        environment.set_state(
            initialization_data=deepcopy(self._initialization_data),
            initialization_actions=deepcopy(self._initialization_actions),
            message_history=deepcopy(history), strict=True,
        )
        return environment

    def observation(self):
        """Public environment context only; never include private task data."""
        return {
            'policy': self._environment.get_policy(),
            'tools': deepcopy([tool.openai_schema for tool in self._environment.get_tools()]),
            'messages': [m.model_dump(mode='json') for m in self._history],
        }

    def step(self, *, call_id, name, arguments):
        """Execute an assistant tool call once and preserve upstream status."""
        from tau2.data_model.message import AssistantMessage, ToolCall

        if not isinstance(call_id, str) or not call_id or call_id in self._call_ids:
            raise ValueError('unique nonempty call ID required')
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError('tool name and argument object required')
        call = ToolCall(id=call_id, name=name, arguments=deepcopy(arguments), requestor='assistant')
        self._call_ids.add(call_id)
        self._history.append(AssistantMessage(role='assistant', tool_calls=[call]))
        response = self._environment.get_response(call)
        self._history.append(response)
        return response.model_copy(deep=True)

    def state_hashes(self):
        """Evaluator-side state evidence; not part of the policy observation."""
        return {'agent': self._environment.get_db_hash(),
                'user': self._environment.get_user_db_hash()}

    def trajectory(self):
        return deepcopy(self._history)

    def replay_hashes(self, history=None):
        environment = self._new_environment(self._history if history is None else history)
        return {'agent': environment.get_db_hash(), 'user': environment.get_user_db_hash()}

    def environment_score(self):
        """Upstream environment component only, not full task/voice success."""
        from tau2.evaluator.evaluator_env import EnvironmentEvaluator

        return EnvironmentEvaluator.calculate_reward(
            environment_constructor=self._constructor, task=self._task,
            full_trajectory=self.trajectory(), strict_replay=True,
        )

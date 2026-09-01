from copy import deepcopy

from maslab import Model, Response


class FixedModel(Model):
    def __init__(self, content='{"vote": "A"}'):
        super().__init__("fixed")
        self.content = content
        self.calls = []

    def respond(self, messages):
        prompt = deepcopy(messages)
        self.calls.append(prompt)
        return Response(
            prompt=prompt,
            content=self.content,
            input_tokens=1,
            output_tokens=1,
            generation_time=0.01,
        )


def hiddenbench_task():
    return {
        "id": "task-1",
        "description": "Choose the supported option.",
        "shared_information": ["Shared fact"],
        "hidden_information": ["Private fact 1", "Private fact 2"],
        "possible_answers": ["A", "B"],
        "correct_answer": "A",
    }

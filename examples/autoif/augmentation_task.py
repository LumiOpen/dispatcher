"""Example task – two responses + judge

run on GPU node with vllm running:
```

# export SSL_CERT_FILE=$(python -m certifi)

module use /appl/local/csc/modulefiles
module load pytorch/2.5

python -m dispatcher.taskmanager.cli \
    --task augmentation_task.InstructionAugmentationTask \
    --input /scratch/project_462000353/adamhrin/dispatcher/examples/autoif/data/categ-v2/aug_input.jsonl \
    --output /scratch/project_462000353/adamhrin/dispatcher/examples/autoif/data/categ-v2/aug_output.jsonl \
    --model /scratch/project_462000353/zosaelai2/models/Llama-3.3-70B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 1 \
    --no-launch \
    --request-timeout 1800
```
"""
from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.backend.request import Request, Response
from dispatcher.taskmanager.task import GeneratorTask, TaskFailed

__all__ = ["InstructionAugmentationTask"]


class InstructionAugmentationTask(GeneratorTask):
    """Generate two answers, have the model judge, and return preferred vs dispreferred."""

    # Fixed generation hyper‑parameters for candidate answers
    GEN_PARAMS: Dict[str, Any] = {
        "temperature": 0.75,
        "top_p": 0.95,
        "max_tokens": 8192,
    }

    # --------------- generator ---------------
    def task_generator(self) -> Generator[Union[Request, List[Request]], Any, Dict[str, Any]]:
        # self.data is prepopulated with the data from the jsonl row being
        # processed
        prompt = self.data.get("prompt")
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response: Response = yield Request({"messages": messages, **self.GEN_PARAMS})
        resp_text = response.get_text()

        # return dict can contain anything you wish to record from this task.
        return {
            "prompt": prompt,
            "original": {"prompt": prompt, "category": self.data.get("category", None)},
            "responses": [resp_text]
        }

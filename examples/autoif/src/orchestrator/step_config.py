"""Pipeline step definitions and configuration"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class SubstepConfig:
    """Configuration for a pipeline substep"""
    name: str
    job_script: str
    input_file_param: Optional[str]  # Parameter name for input file
    output_file_param: str  # Parameter name for output file
    requires_gpu: bool
    default_time: str  # Default SLURM time limit
    default_nodes: int
    default_ntasks_per_node: int
    description: str

    def get_input_file(self, params: Dict[str, Any]) -> Optional[str]:
        """Get input file path from parameters"""
        if not self.input_file_param:
            return None
        return params.get(self.input_file_param)

    def get_output_file(self, params: Dict[str, Any]) -> str:
        """Get output file path from parameters"""
        return params[self.output_file_param]


@dataclass
class StepConfig:
    """Configuration for a pipeline step (may contain multiple substeps)"""
    name: str
    substeps: List[SubstepConfig]
    description: str

    def get_substep(self, substep_name: str) -> Optional[SubstepConfig]:
        """Get a specific substep by name"""
        for substep in self.substeps:
            if substep.name == substep_name:
                return substep
        return None


# Define all pipeline steps with their substeps
STEP_DEFINITIONS = {
    "augmentation": StepConfig(
        name="augmentation",
        description="Augment seed instructions using LLM",
        substeps=[
            SubstepConfig(
                name="augmentation_job",
                job_script="jobs/augmentation_job.sh",
                input_file_param="seed_file",
                output_file_param="output_file",
                requires_gpu=True,
                default_time="00:30:00",
                default_nodes=1,
                default_ntasks_per_node=2,
                description="GPU: Generate augmented instructions"
            ),
            SubstepConfig(
                name="augmentation_postprocessing",
                job_script="jobs/augmentation_postprocessing.sh",
                input_file_param="output_file",
                output_file_param="augmented_instructions_file",
                requires_gpu=False,
                default_time="00:15:00",
                default_nodes=1,
                default_ntasks_per_node=1,
                description="CPU: Filter and deduplicate instructions"
            )
        ]
    ),

    "verifiers": StepConfig(
        name="verifiers",
        description="Generate and validate verification functions",
        substeps=[
            SubstepConfig(
                name="verifiers_preprocessing",
                job_script="jobs/verifiers_preprocessing.sh",
                input_file_param="augmented_instructions_file",
                output_file_param="verifiers_input_file",
                requires_gpu=False,
                default_time="00:10:00",
                default_nodes=1,
                default_ntasks_per_node=1,
                description="CPU: Create verifier prompts"
            ),
            SubstepConfig(
                name="verifiers_job",
                job_script="jobs/verifiers_job.sh",
                input_file_param="verifiers_input_file",
                output_file_param="output_file",
                requires_gpu=True,
                default_time="02:00:00",
                default_nodes=1,
                default_ntasks_per_node=2,
                description="GPU: Generate verification functions"
            ),
            SubstepConfig(
                name="verifiers_postprocessing",
                job_script="jobs/verifiers_postprocessing.sh",
                input_file_param="output_file",
                output_file_param="verifiers_filtered_file",
                requires_gpu=False,
                default_time="01:00:00",
                default_nodes=1,
                default_ntasks_per_node=1,
                description="CPU: Cross-validate verifiers"
            )
        ]
    ),

    "concatenation": StepConfig(
        name="concatenation",
        description="Concatenate queries with instructions",
        substeps=[
            SubstepConfig(
                name="concatenation_job",
                job_script="jobs/concatenation_job.sh",
                input_file_param="verifiers_file",
                output_file_param="output_file",
                requires_gpu=False,
                default_time="01:00:00",
                default_nodes=1,
                default_ntasks_per_node=1,
                description="CPU: Create query dataset"
            )
        ]
    ),

    "responses": StepConfig(
        name="responses",
        description="Generate and score query responses",
        substeps=[
            SubstepConfig(
                name="responses_job",
                job_script="jobs/response_job.sh",
                input_file_param="input_file",
                output_file_param="output_file",
                requires_gpu=True,
                default_time="18:00:00",
                default_nodes=4,
                default_ntasks_per_node=2,
                description="GPU: Generate and score responses"
            )
        ]
    ),

    "sft": StepConfig(
        name="sft",
        description="Build final SFT dataset",
        substeps=[
            SubstepConfig(
                name="sft_job",
                job_script="jobs/final_dataset_job.sh",
                input_file_param="input_file",
                output_file_param="output_dir",
                requires_gpu=False,
                default_time="00:30:00",
                default_nodes=1,
                default_ntasks_per_node=1,
                description="CPU: Build SFT dataset"
            )
        ]
    )
}


def get_step_config(step_name: str) -> StepConfig:
    """
    Get configuration for a pipeline step.

    Args:
        step_name: Name of the step

    Returns:
        StepConfig for the step

    Raises:
        KeyError if step not found
    """
    if step_name not in STEP_DEFINITIONS:
        raise KeyError(f"Unknown step: {step_name}")
    return STEP_DEFINITIONS[step_name]


def get_substep_config(step_name: str, substep_name: str) -> SubstepConfig:
    """
    Get configuration for a specific substep.

    Args:
        step_name: Name of the parent step
        substep_name: Name of the substep

    Returns:
        SubstepConfig for the substep

    Raises:
        KeyError if step or substep not found
    """
    step = get_step_config(step_name)
    substep = step.get_substep(substep_name)
    if not substep:
        raise KeyError(f"Unknown substep '{substep_name}' in step '{step_name}'")
    return substep


def get_all_steps() -> List[str]:
    """Get list of all step names in order"""
    return list(STEP_DEFINITIONS.keys())


def get_step_order() -> List[str]:
    """Get canonical order of steps for execution"""
    return [
        "augmentation",
        "verifiers",
        "concatenation",
        "responses",
        "sft"
    ]


def get_substeps_for_step(step_name: str) -> List[str]:
    """
    Get list of substep names for a step.

    Args:
        step_name: Name of the step

    Returns:
        List of substep names
    """
    step = get_step_config(step_name)
    return [substep.name for substep in step.substeps]

"""
Gen2 QwenImage Core - Scheduler Utilities

VideoX FlowMatch scheduler creation and utility functions.
"""

import inspect

from .imports import FlowMatchEulerDiscreteScheduler


def filter_kwargs(cls, kwargs):
    """Filter kwargs to only include parameters accepted by cls.__init__"""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_qwen_scheduler(sampler_name: str, shift: float):
    """
    Create a FlowMatch scheduler matching VideoX's get_qwen_scheduler.
    
    Args:
        sampler_name: One of "Flow", "Flow_Unipc", "Flow_DPM++"
        shift: Shift parameter for the scheduler
    
    Returns:
        Configured scheduler instance
    """
    # Try to import VideoX's custom schedulers, fall back to standard FlowMatch
    try:
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        HAS_VIDEOX_SCHEDULERS = True
    except ImportError:
        HAS_VIDEOX_SCHEDULERS = False
    
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler if HAS_VIDEOX_SCHEDULERS else FlowMatchEulerDiscreteScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler if HAS_VIDEOX_SCHEDULERS else FlowMatchEulerDiscreteScheduler,
    }[sampler_name]
    
    # Match VideoX's scheduler defaults
    scheduler_kwargs = {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler_kwargs["shift"] = shift
    scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, scheduler_kwargs))
    return scheduler


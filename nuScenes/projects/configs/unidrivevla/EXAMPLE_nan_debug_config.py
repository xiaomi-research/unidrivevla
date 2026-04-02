"""
NaN Debugger Configuration Example

Add this to your training config file to enable NaN debugging:

1. Add NaN debugger configuration:
   nan_debugger = dict(
       enabled=True  # Set to False to disable
   )

2. Add NaN debug hook to custom_hooks:
   custom_hooks = [
       dict(
           type='NaNDebugHook',
           check_losses=True,        # Check loss values
           check_gradients=True,     # Check gradients
           check_parameters=False,   # Check parameters (slow, usually not needed)
           check_params_interval=100,
           verbose=True,             # Print detailed gradient statistics
           abort_on_nan=True,        # Abort training if NaN detected
           priority='VERY_HIGH'      # Run this hook first
       )
   ]

Full example:
"""

_base_ = ['./unified_decoder_stage1.py']

# Enable NaN debugger
nan_debugger = dict(
    enabled=True
)

# Add NaN debug hook
custom_hooks = [
    dict(
        type='NaNDebugHook',
        check_losses=True,
        check_gradients=True,
        check_parameters=False,
        check_params_interval=100,
        verbose=True,  # Set to False for less output
        abort_on_nan=True,
        priority='VERY_HIGH'
    )
]

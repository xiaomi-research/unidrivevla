#!/usr/bin/env python3
"""
Script to fix Python 3.8 compatibility issues in transformers library.
Replaces torch.Tensor with "torch.Tensor" in type annotations to avoid
TypeError: 'ABCMeta' object is not subscriptable error.
"""

import os
import re
import sys
from pathlib import Path

# Patterns to search for and fix
PATTERNS = [
    # Tuple["torch.Tensor", ...]
    (r'Tuple\[torch\.Tensor', r'Tuple["torch.Tensor"'),
    # Optional["torch.Tensor"]
    (r'Optional\[torch\.Tensor', r'Optional["torch.Tensor"'),
    # List["torch.Tensor"]
    (r'List\[torch\.Tensor', r'List["torch.Tensor"'),
    # Dict[..., "torch.Tensor"] - need to be careful with this one
    (r'Dict\[(.*?),\s*torch\.Tensor', r'Dict[\1, "torch.Tensor"'),
    # Union[..., "torch.Tensor"]
    (r'Union\[([^,]*?),\s*torch\.Tensor', r'Union[\1, "torch.Tensor"'),
    # Union["torch.Tensor", ...]
    (r'Union\[torch\.Tensor', r'Union["torch.Tensor"'),
    # Function parameter: torch.Tensor (without any wrapper)
    # This pattern matches "param_name: torch.Tensor" or "param_name: "torch.Tensor" ="
    (r'(\w+):\s*torch\.Tensor(\s*[=\)]|,)', r'\1: "torch.Tensor"\2'),
]

def fix_file(file_path):
    """Fix torch.Tensor type annotations in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all patterns
        for pattern, replacement in PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        # Only write if there were changes
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all files in the transformers directory."""
    transformers_dir = Path("/high_perf_store3/world-model/yongkangli/ABCDEFG_NISHIDASHABI/A/B/UniDriveVLA/Bench2Drive/transformers-4.57.1")
    
    if not transformers_dir.exists():
        print(f"Directory not found: {transformers_dir}")
        sys.exit(1)
    
    # Patterns to match Python files
    python_patterns = ["*.py", "*.pyi"]
    
    files_fixed = 0
    files_processed = 0
    
    print(f"Scanning directory: {transformers_dir}")
    
    for pattern in python_patterns:
        for file_path in transformers_dir.rglob(pattern):
            # Skip certain directories that might not need fixing
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'build', 'dist']):
                continue
            
            files_processed += 1
            if files_processed % 100 == 0:
                print(f"Processed {files_processed} files...")
            
            if fix_file(file_path):
                files_fixed += 1
                print(f"Fixed: {file_path.relative_to(transformers_dir)}")
    
    print(f"\nSummary:")
    print(f"Files processed: {files_processed}")
    print(f"Files fixed: {files_fixed}")
    print(f"Python 3.8 compatibility fixes applied successfully!")

if __name__ == "__main__":
    main()
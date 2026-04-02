#!/usr/bin/env python3
"""
Script to fix Python 3.8 compatibility issues in transformers library.
Replaces subscripted built-in types with typing equivalents.
"""

import os
import re
from pathlib import Path
from typing import List, Set, Tuple, Dict, FrozenSet


def find_python_files_with_subscripted_types(base_path: str) -> List[str]:
    """Find all Python files that use subscripted built-in types."""
    files_with_issues = []
    
    # Patterns to search for (escaped properly)
    patterns = [
        r'tuple\[',  # Tuple[...]
        r'dict\[',   # Dict[...]
        r'list\[',   # List[...]
        r'set\[',    # Set[...]
        r'frozenset\['  # frozenSet[...]
    ]
    
    base_dir = Path(base_path)
    
    for py_file in base_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Check if file contains any of the problematic patterns
            has_issue = any(re.search(pattern, content) for pattern in patterns)
            
            if has_issue:
                files_with_issues.append(str(py_file))
                
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
            return []
    
    return files_with_issues


def fix_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    Fix a single file by replacing subscripted built-in types with typing equivalents.
    Returns (success, list_of_changes)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # Define replacements - use raw strings to avoid issues
        replacements = [
            (r'tuple\[', 'Tuple['),
            (r'dict\[', 'Dict['),
            (r'list\[', 'List['),
            (r'set\[', 'Set['),
            (r'frozenset\[', 'FrozenSet['),
        ]
        
        # Apply replacements
        for pattern, replacement in replacements:
            # Count occurrences before replacement
            count_before = len(re.findall(pattern, content))
            if count_before > 0:
                content = re.sub(pattern, replacement, content)
                count_after = len(re.findall(replacement, content))
                changes.append(f"Replaced {pattern} with {replacement} ({count_before} occurrences)")
        
        # Check if we need to add imports
        needs_imports = False
        imports_to_add = []
        
        # Check which typing imports are needed
        if re.search(r'Tuple\[', content) and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('Tuple')
        
        if re.search(r'Dict\[', content) and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('Dict')
        
        if re.search(r'List\[', content) and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('List')
        
        if re.search(r'Set\[', content) and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('Set')
        
        if re.search(r'FrozenSet\[', content) and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('FrozenSet')
        
        # Add imports if needed
        if needs_imports and imports_to_add:
            # Find the first import statement
            import_match = re.search(r'^(from \w+ import|import \w+)', content, re.MULTILINE)
            
            if import_match:
                # Insert imports after the first import
                import_pos = import_match.end()
                import_line = import_match.group(0)
                
                # Check if there's already a typing import
                typing_import_match = re.search(r'from typing import', content)
                
                if typing_import_match:
                    # Add to existing typing import
                    existing_import = typing_import_match.group(0)
                    # Extract current imports
                    current_imports = re.search(r'from typing import (.+)', content)
                    if current_imports:
                        current_list = current_imports.group(1)
                        # Add new imports
                        new_imports = set(current_list.split(', ') + imports_to_add)
                        new_import_line = f"from typing import {', '.join(sorted(new_imports))}"
                        content = content.replace(existing_import, new_import_line)
                        changes.append(f"Updated typing import to include: {', '.join(imports_to_add)}")
                else:
                    # Add new typing import
                    import_line_end = content.find('\n', import_pos)
                    if import_line_end == -1:
                        import_line_end = len(content)
                    
                    new_import = f"from typing import {', '.join(sorted(imports_to_add))}\n"
                    content = content[:import_line_end] + new_import + content[import_line_end:]
                    changes.append(f"Added typing import: {', '.join(imports_to_add)}")
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        else:
            return False, []
            
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False, [f"Error: {e}"]


def main():
    """Main function to fix all files."""
    base_path = "/high_perf_store3/world-model/yongkangli/ABCDEFG_NISHIDASHABI/A/B/UniDriveVLA/Bench2Drive/transformers-4.57.1/src/transformers"
    
    print("Searching for Python files with subscripted built-in types...")
    files_with_issues = find_python_files_with_subscripted_types(base_path)
    
    print(f"Found {len(files_with_issues)} files with issues")
    
    if not files_with_issues:
        print("No files need fixing!")
        return
    
    print("\nFixing files...")
    
    fixed_files = []
    failed_files = []
    
    for i, file_path in enumerate(files_with_issues, 1):
        print(f"[{i}/{len(files_with_issues)}] Fixing: {os.path.basename(file_path)}")
        
        success, changes = fix_file(file_path)
        
        if success:
            fixed_files.append((file_path, changes))
            if changes:
                print(f"  ✓ Fixed with changes: {changes[0]}")
        else:
            failed_files.append((file_path, changes))
            print(f"  ✗ Failed: {changes}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {len(files_with_issues)}")
    print(f"Successfully fixed: {len(fixed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    if fixed_files:
        print(f"\nFixed files:")
        for file_path, changes in fixed_files:
            print(f"  - {file_path}")
            for change in changes:
                print(f"    {change}")
    
    if failed_files:
        print(f"\nFailed files:")
        for file_path, errors in failed_files:
            print(f"  - {file_path}: {errors}")


if __name__ == "__main__":
    main()
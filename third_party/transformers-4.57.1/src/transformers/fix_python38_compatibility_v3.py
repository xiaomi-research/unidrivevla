#!/usr/bin/env python3
"""
Script to fix Python 3.8 compatibility issues in transformers library.
Replaces subscripted built-in types with typing equivalents.
"""

import os
from pathlib import Path
from typing import List, Set, Tuple, Dict, FrozenSet


def find_python_files_with_subscripted_types(base_path: str) -> List[str]:
    """Find all Python files that use subscripted built-in types."""
    files_with_issues = []
    
    base_dir = Path(base_path)
    
    for py_file in base_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Check if file contains any of the problematic patterns
            has_issue = (
                'Tuple[' in content or
                'Dict[' in content or
                'List[' in content or
                'Set[' in content or
                'frozenSet[' in content
            )
            
            if has_issue:
                files_with_issues.append(str(py_file))
                
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
            continue
    
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
        
        # Define replacements
        replacements = [
            ('Tuple[', 'Tuple['),
            ('Dict[', 'Dict['),
            ('List[', 'List['),
            ('Set[', 'Set['),
            ('frozenSet[', 'FrozenSet['),
        ]
        
        # Apply replacements
        for old_str, new_str in replacements:
            count_before = content.count(old_str)
            if count_before > 0:
                content = content.replace(old_str, new_str)
                count_after = content.count(new_str)
                changes.append(f"Replaced {old_str} with {new_str} ({count_before} occurrences)")
        
        # Check if we need to add imports
        needs_imports = False
        imports_to_add = []
        
        # Check which typing imports are needed
        if 'Tuple[' in content and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('Tuple')
        
        if 'Dict[' in content and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('Dict')
        
        if 'List[' in content and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('List')
        
        if 'Set[' in content and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('Set')
        
        if 'FrozenSet[' in content and 'from typing import' not in content and 'import typing' not in content:
            needs_imports = True
            imports_to_add.append('FrozenSet')
        
        # Add imports if needed
        if needs_imports and imports_to_add:
            # Find the first import statement
            lines = content.split('\n')
            import_line_idx = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith(('from ', 'import ')):
                    import_line_idx = i
                    break
            
            if import_line_idx >= 0:
                # Check if there's already a typing import
                typing_import_idx = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith('from typing import'):
                        typing_import_idx = i
                        break
                
                if typing_import_idx >= 0:
                    # Add to existing typing import
                    existing_line = lines[typing_import_idx]
                    # Extract current imports
                    if 'from typing import' in existing_line:
                        # Get the part after "from typing import"
                        import_part = existing_line.split('from typing import')[1].strip()
                        # Parse existing imports
                        existing_imports = [imp.strip() for imp in import_part.split(',')]
                        # Add new imports
                        all_imports = set(existing_imports + imports_to_add)
                        new_line = f"from typing import {', '.join(sorted(all_imports))}"
                        lines[typing_import_idx] = new_line
                        changes.append(f"Updated typing import to include: {', '.join(imports_to_add)}")
                else:
                    # Add new typing import after the first import
                    new_import_line = f"from typing import {', '.join(sorted(imports_to_add))}"
                    lines.insert(import_line_idx + 1, new_import_line)
                    changes.append(f"Added typing import: {', '.join(imports_to_add)}")
                
                content = '\n'.join(lines)
        
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
#!/usr/bin/env python3
"""
Bulk import fixer for calipod refactoring.
Replaces all old-style imports (calipod.logger, calipod.packets, etc.)
with new-style imports (calipod.core.logger, calipod.core.packets, etc.)

Run this script from the Calipod project root:
    python fix_imports.py
"""

import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a single Python file."""
    content = file_path.read_text(encoding='utf-8')
    original_content = content
    
    replacements = [
        # Import statements (these go first)
        (r'import calipod\.logger(?!\w)', 'from calipod.core import logger as calipod_logger'),
        (r'from calipod\.logger import', 'from calipod.core.logger import'),
        (r'from calipod\.packets import', 'from calipod.core.packets import'),
        (r'from calipod\.helper import', 'from calipod.core.helper import'),
        (r'from calipod\.configurator import', 'from calipod.core.configurator import'),
        (r'from calipod\.controller import', 'from calipod.core.controller import'),
        
        # Usage patterns (these go second)
        (r'calipod\.logger\.get\(', 'calipod_logger.get('),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """Fix all Python files in calipod and tests directories."""
    project_root = Path.cwd()
    
    # Find all Python files
    python_files = list(project_root.glob('calipod/**/*.py')) + list(project_root.glob('tests/**/*.py'))
    
    fixed_count = 0
    
    for py_file in python_files:
        try:
            if fix_imports_in_file(py_file):
                print(f"✓ Fixed: {py_file.relative_to(project_root)}")
                fixed_count += 1
        except Exception as e:
            print(f"⚠ Error in {py_file}: {e}")
    
    print(f"\n✓ Fixed {fixed_count} files")

if __name__ == '__main__':
    main()

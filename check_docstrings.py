#!/usr/bin/env python3
"""Script to check for Russian text in docstrings within tensoraerospace directory.

This script scans all Python files in the tensoraerospace directory and reports
any docstrings containing Cyrillic characters.
"""

import ast
import re
from pathlib import Path
from typing import List, Tuple


def contains_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters.

    Args:
        text: Text to check.

    Returns:
        bool: True if text contains Cyrillic characters, False otherwise.
    """
    return bool(re.search(r'[А-Яа-яЁё]', text))


def extract_docstrings(node: ast.AST) -> List[str]:
    """Extract all docstrings from an AST node.

    Args:
        node: AST node to extract docstrings from.

    Returns:
        List of docstring strings found in the node.
    """
    docstrings = []
    
    # Check if node has a docstring
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
        if ast.get_docstring(node):
            docstrings.append(ast.get_docstring(node))
    
    # Recursively check child nodes
    for child in ast.iter_child_nodes(node):
        docstrings.extend(extract_docstrings(child))
    
    return docstrings


def check_file(file_path: Path) -> List[Tuple[int, str]]:
    """Check a Python file for Russian docstrings.

    Args:
        file_path: Path to the Python file.

    Returns:
        List of tuples (line_number, docstring_content) containing Russian text.
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Parse AST
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            print(f"Warning: Syntax error in {file_path}: {e}")
            return issues
        
        # Extract docstrings
        docstrings = extract_docstrings(tree)
        
        # Check each docstring for Cyrillic
        for docstring in docstrings:
            if docstring and contains_cyrillic(docstring):
                # Find line number of the docstring
                line_num = content.find(docstring.split('\n')[0])
                if line_num != -1:
                    line_number = content[:line_num].count('\n') + 1
                else:
                    line_number = 0
                
                # Extract first few lines for context
                preview = '\n'.join(docstring.split('\n')[:5])
                if len(docstring.split('\n')) > 5:
                    preview += '\n...'
                
                issues.append((line_number, preview))
        
        # Also check for docstrings in comments (commented out docstrings)
        commented_issues = []
        for i, line in enumerate(lines, 1):
            # Check for commented docstrings
            if re.search(r'#\s*"""[^"]*[А-Яа-яЁё]', line):
                commented_issues.append((i, f"Commented docstring: {line.strip()}"))
        
        return issues, commented_issues
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return issues


def check_directory(directory: Path) -> tuple:
    """Check all Python files in a directory for Russian docstrings.

    Args:
        directory: Path to directory to check.

    Returns:
        Tuple of (active_issues_dict, commented_issues_dict).
    """
    active_results = {}
    commented_results = {}
    
    # Find all Python files recursively
    python_files = list(directory.rglob('*.py'))
    
    print(f"Checking {len(python_files)} Python files in {directory}...")
    print("-" * 80)
    
    for file_path in sorted(python_files):
        # Skip __pycache__ directories
        if '__pycache__' in str(file_path):
            continue
        
        issues, commented_issues = check_file(file_path)
        if issues:
            active_results[str(file_path)] = issues
        if commented_issues:
            commented_results[str(file_path)] = commented_issues
    
    return active_results, commented_results


def print_report(active_results: dict, commented_results: dict) -> None:
    """Print a formatted report of findings.

    Args:
        active_results: Dictionary mapping file paths to lists of active docstring issues.
        commented_results: Dictionary mapping file paths to lists of commented docstring issues.
    """
    total_active = sum(len(issues) for issues in active_results.values())
    total_commented = sum(len(issues) for issues in commented_results.values())
    
    if not active_results and not commented_results:
        print("\n✓ No Russian text found in docstrings!")
        print("All docstrings are properly translated to English.")
        return
    
    if active_results:
        print(f"\n✗ Found Russian text in ACTIVE docstrings in {len(active_results)} file(s):\n")
        
        for file_path, issues in active_results.items():
            print(f"\nFile: {file_path}")
            print("-" * 80)
            for line_num, docstring in issues:
                print(f"  Line {line_num}:")
                print(f"    {docstring}")
                print()
        
        print(f"\nTotal ACTIVE issues: {total_active}")
        print(f"Files with ACTIVE issues: {len(active_results)}")
    
    if commented_results:
        print(f"\n⚠ Found Russian text in COMMENTED docstrings in {len(commented_results)} file(s):")
        print("(These are commented out and don't need translation)\n")
        
        for file_path, issues in commented_results.items():
            print(f"  File: {file_path}")
            for line_num, docstring in issues:
                print(f"    Line {line_num}: {docstring.split(':')[1] if ':' in docstring else docstring}")
        
        print(f"\nTotal COMMENTED issues: {total_commented}")
        print(f"Files with COMMENTED issues: {len(commented_results)}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Active docstrings with Russian: {total_active} in {len(active_results)} file(s)")
    print(f"  Commented docstrings with Russian: {total_commented} in {len(commented_results)} file(s)")
    print(f"  {'✓ All active docstrings translated!' if total_active == 0 else '✗ Translation needed!'}")


def main():
    """Main function to run the docstring checker."""
    import sys
    
    # Get directory path from command line or use default
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        # Default to tensoraerospace directory
        script_dir = Path(__file__).parent
        directory = script_dir / 'tensoraerospace'
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist!")
        sys.exit(1)
    
    print(f"Checking docstrings in: {directory}")
    print("=" * 80)
    
    active_results, commented_results = check_directory(directory)
    print_report(active_results, commented_results)
    
    # Exit with error code if active issues found (commented ones don't count)
    if active_results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()


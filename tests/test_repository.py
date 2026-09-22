"""CPU-only release checks; no weights, torch import or GPU needed."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_syntax_and_local_reference_imports():
    vendor = ROOT / 'flashloop_torch/_reference'
    names = {p.stem for p in vendor.glob('*.py')}
    for path in ROOT.rglob('*.py'):
        if any(part in {'.venv', 'build', 'venv'} for part in path.parts):
            continue
        tree = ast.parse(path.read_text())
        if path.parent == vendor:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in names:
                    assert node.level == 1, (path, node.module)
                if isinstance(node, ast.ImportFrom) and node.level == 1:
                    assert node.module in names, (path, node.module)


def test_cli_help_does_not_load_model():
    result = subprocess.run([sys.executable, str(ROOT/'examples/generate.py'), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--backend' in result.stdout and '--model' in result.stdout


def test_backend_attention_contract():
    spec = importlib.util.spec_from_file_location('generate', ROOT/'examples/generate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.attention_implementation('torch') == 'eager'
    assert module.attention_implementation('engine') == 'sdpa'


def test_smoke_requires_explicit_model():
    result = subprocess.run(['bash', str(ROOT/'scripts/smoke_test.sh')],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Usage:' in result.stderr

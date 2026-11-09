"""Read-only inventory of the Flask/Jinja frontend contract."""
import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Fields(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if any(key in attrs for key in ("id", "name")) or any(key.startswith('data-') for key in attrs):
            self.fields.append({"tag": tag, **attrs})


def inventory():
    result = {"routes": [], "templates": {}, "javascript": {}, "training_keys": {}, "json_keys": {}, "source_templates": []}
    sources = list(ROOT.glob("app*.py")) + [ROOT / 'auth_module.py', ROOT / 'modules/compatibility.py']
    for folder in ('problems', 'policies', 'algorithms', 'rl_training'):
        sources.extend((ROOT / 'modules' / folder).rglob('*.py'))
    for path in sorted(sources):
        source = path.read_text(encoding="utf-8-sig")
        key = path.relative_to(ROOT).as_posix()
        result["training_keys"][key] = sorted(set(re.findall(r"(?:config|data|results)\.get\(['\"]([^'\"]+)", source)))
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            result['source_templates'].append({'file': key, 'line': error.lineno, 'reason': error.msg})
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "route":
                        result["routes"].append({"file": path.name, "function": node.name, "route": ast.literal_eval(deco.args[0]), "methods": next((ast.literal_eval(k.value) for k in deco.keywords if k.arg == "methods"), ["GET"])})
        result['json_keys'][key] = sorted({item.value for node in ast.walk(tree) if isinstance(node, ast.Dict) for item in node.keys if isinstance(item, ast.Constant) and isinstance(item.value, str)})
    for path in sorted((ROOT / "templates").rglob("*.html")):
        parser = Fields()
        parser.feed(path.read_text(encoding="utf-8-sig"))
        result["templates"][path.relative_to(ROOT).as_posix()] = parser.fields
    for path in sorted((ROOT / "static/js").glob("*.js")):
        source = path.read_text(encoding="utf-8-sig")
        result["javascript"][path.relative_to(ROOT).as_posix()] = {
            "ids": sorted(set(re.findall(r"getElementById\(['\"]([^'\"]+)", source))),
            "selectors": sorted(set(re.findall(r"querySelector(?:All)?\(['\"]([^'\"]+)", source))),
            "requests": sorted(set(re.findall(r"(?:fetch|EventSource|Studio\.request|Studio\.post)\(\s*['\"`]([^'\"`]+)", source))),
        }
    return result


if __name__ == "__main__":
    import sys
    output = json.dumps(inventory(), ensure_ascii=False, indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(output, encoding="utf-8")
    else:
        print(output)

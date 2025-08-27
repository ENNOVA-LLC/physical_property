import argparse, pathlib, sys, libcst as cst
from typing import List, Tuple, Optional, Dict

LOG_FUNCS = {"debug","info","warning","error","exception","critical"}

def _esc_percent(s: str) -> str:
    return s.replace("%", "%%")

def _which_log_call(node: cst.Call) -> Tuple[str, int] | None:
    func = node.func
    name = None
    if isinstance(func, cst.Attribute) and isinstance(func.attr, cst.Name):
        name = func.attr.value
    elif isinstance(func, cst.Name):
        name = func.value
    if name in LOG_FUNCS and len(node.args) >= 1:
        return (name, 0)
    return ("log", 1) if name == "log" and len(node.args) >= 2 else None

class Scope:
    def __init__(self):
        self.msgs: Dict[str, cst.FormattedString] = {}

class Rewriter(cst.CSTTransformer):
    def __init__(self, path: str, verbose: bool = False):
        self.path = path
        self.verbose = verbose
        self.changed = False
        self.checks: List[str] = []
        self.scopes: List[Scope] = [Scope()]

    def visit_FunctionDef(self, node): self.scopes.append(Scope())
    def leave_FunctionDef(self, orig, upd): self.scopes.pop(); return upd
    def visit_ClassDef(self, node): self.scopes.append(Scope())
    def leave_ClassDef(self, orig, upd): self.scopes.pop(); return upd
    def _scope(self) -> Scope: return self.scopes[-1]

    # capture: msg = f"..."
    def visit_Assign(self, node: cst.Assign):
        if len(node.targets) == 1 and isinstance(node.targets[0].target, cst.Name):
            if isinstance(node.value, cst.FormattedString):
                self._scope().msgs[node.targets[0].target.value] = node.value

    def _convert_fstring(self, fs: cst.FormattedString):
        parts: List[str] = []
        args: List[cst.Arg] = []
        needs_review = False
        for p in fs.parts:
            if isinstance(p, cst.FormattedStringText):
                parts.append(_esc_percent(p.value))
            elif isinstance(p, cst.FormattedStringExpression):
                # map !r/!s/!a to %r/%s (approximate !a as %r)
                placeholder = "%s"
                conv = getattr(p, "conversion", None)
                conv_val = getattr(conv, "value", None) if conv is not None else None
                if conv_val == "r" or conv_val == "a":
                    placeholder = "%r"
                elif conv_val == "s":
                    placeholder = "%s"
                # format specs (e.g., :.3f) need manual review
                if getattr(p, "format_spec", None) is not None:
                    needs_review = True
                parts.append(placeholder)
                args.append(cst.Arg(p.expression))  # <-- FIX: use .expression
        fmt = cst.SimpleString('"' + "".join(parts) + '"')
        return fmt, args, needs_review

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call):
        hit = _which_log_call(original_node)
        if not hit:
            return updated_node
        _, msg_idx = hit

        msg_node = original_node.args[msg_idx].value
        fs: Optional[cst.FormattedString] = None
        if isinstance(msg_node, cst.FormattedString):
            fs = msg_node
        elif isinstance(msg_node, cst.Name):
            fs = self._scope().msgs.get(msg_node.value)
        if fs is None:
            return updated_node

        fmt, new_args, needs_review = self._convert_fstring(fs)
        args = list(updated_node.args)
        args[msg_idx] = cst.Arg(fmt)
        args[msg_idx+1:msg_idx+1] = new_args
        node = updated_node.with_changes(args=args)

        self.changed = True
        if self.verbose:
            print(f"rewrite: {self.path}", file=sys.stderr)
        if needs_review:
            self.checks.append(f"{self.path}: dropped format spec -> review")
        return node

def iter_py_files(root: pathlib.Path):
    if root.is_file() and root.suffix == ".py": yield root
    elif root.is_dir(): yield from root.glob("**/*.py")

def run(paths: List[str], dry_run: bool, verbose: bool, list_files: bool):
    total, changed = 0, 0
    checks: List[str] = []
    for root in paths:
        for py in iter_py_files(pathlib.Path(root)):
            total += 1
            if list_files: print(py)
            try:
                mod = cst.parse_module(py.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"skip parse error: {py}: {e}", file=sys.stderr)
                continue
            rw = Rewriter(str(py), verbose=verbose)
            new = mod.visit(rw)
            checks.extend(rw.checks)
            if rw.changed:
                changed += 1
                if not dry_run:
                    py.write_text(new.code, encoding="utf-8")
    print(f"[summary] scanned={total} changed={changed} dry_run={dry_run}")
    for m in checks:
        print("CHECK:", m, file=sys.stderr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    run(a.paths, dry_run=a.dry_run, verbose=a.verbose, list_files=a.list)

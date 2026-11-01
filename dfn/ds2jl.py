"""Translate the DiffSL subset emitted by pybamm.DiffSLExport into Julia.

uv run dfn/ds2jl.py dfn/dfn.ds julia/dfn_gen.jl

Generates: sparse/dense constants, `dfn_eval(u, p, t) -> (F, out)`, `dfn_rhs!(du, u, p, t)`,
`dfn_mass` (diagonal, 0 on algebraic rows), `dfn_u0(p)`, `DFN_NSTATES`, `DFN_NPARAMS`.
Anything outside the subset raises, rather than translating silently wrong.
"""

import re
import sys

# ponytail: allocating out-of-place translation (simple and AD-friendly); generate in-place
# code with caches if the Julia RHS cost dominates the DFN timings.

FUNCS1 = {"exp", "log", "sqrt", "abs", "sin", "cos", "tan", "tanh", "sinh", "cosh", "arcsinh"}
FUNCS2 = {"max", "min", "copysign"}
TOKEN = re.compile(r"\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?)|([A-Za-z][A-Za-z0-9]*(?:_[a-z]+)?)|(.))")


def tokenize(s):
    out = []
    for num, name, op in TOKEN.findall(s):
        out.append(("num", num) if num else ("name", name) if name else ("op", op))
    return [t for t in out if t != ("op", "")]


class Parser:
    """expr := term (+|- term)* ; term := unary (*|/ unary)* ; unary := -unary | primary"""

    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, val=None):
        tok = self.peek()
        if val is not None and tok[1] != val:
            raise SyntaxError(f"expected {val!r}, got {tok}")
        self.i += 1
        return tok

    def expr(self):
        node = self.term()
        while self.peek()[1] in ("+", "-"):
            node = (self.take()[1], node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek()[1] in ("*", "/"):
            node = (self.take()[1], node, self.unary())
        return node

    def unary(self):
        if self.peek()[1] == "-":
            self.take()
            return ("neg", self.unary())
        return self.primary()

    def primary(self):
        kind, val = self.take()
        if kind == "num":
            return ("num", val)
        if val == "(":
            node = self.expr()
            self.take(")")
            return node
        if kind == "name":
            if self.peek()[1] == "(":
                self.take("(")
                args = [self.expr()]
                while self.peek()[1] == ",":
                    self.take(",")
                    args.append(self.expr())
                self.take(")")
                return ("call", val, args)
            if self.peek()[1] == "[":
                self.take("[")
                a = int(self.take()[1])
                self.take(":")
                b = int(self.take()[1])
                self.take("]")
                return ("slice", val, a, b)
            return ("var", val)
        raise SyntaxError(f"unexpected {val!r}")


def base(name):
    return name.split("_")[0]


def is_matrix(node):
    return node[0] == "var" and node[1].endswith("_ij")


def emit(node, inputs):
    kind = node[0]
    if kind == "num":
        return f"{float(node[1])!r}"
    if kind == "var":
        name = node[1]
        if name == "t":
            return "t"
        if name in inputs:
            return name
        return base(name)
    if kind == "slice":
        return f"{base(node[1])}[{node[2] + 1}:{node[3]}]"
    if kind == "neg":
        return f"(.-{emit(node[1], inputs)})"
    if kind == "call":
        f, args = node[1], [emit(a, inputs) for a in node[2]]
        if f == "pow":
            return f"(({args[0]}) .^ ({args[1]}))"
        if f == "heaviside":
            return f"ifelse.(({args[0]}) .>= 0, 1.0, 0.0)"
        if f == "arcsinh":
            return f"asinh.({args[0]})"
        if f in FUNCS1 and len(args) == 1 or f in FUNCS2 and len(args) == 2:
            return f"{f}.({', '.join(args)})"
        raise SyntaxError(f"unsupported function {f}/{len(args)}")
    op, lhs, rhs = node
    if op == "*" and is_matrix(lhs):
        return f"({base(lhs[1])} * ({emit(rhs, inputs)}))"
    if is_matrix(lhs) or is_matrix(rhs):
        raise SyntaxError("matrix outside a mat-vec")
    return f"(({emit(lhs, inputs)}) .{op} ({emit(rhs, inputs)}))"


ENTRY = re.compile(r"^\s*(?:\((\d+)(?:\.\.(\d+))?\s*(?::\s*(\d+))?\s*(?:,\s*(\d+)(?:\.\.(\d+))?\s*)?\)\s*:)?\s*(?:([A-Za-z][A-Za-z0-9]*)\s*=)?\s*(.*?)\s*$")


def blocks(text):
    """[(name, index, [entry strings])] in file order; entries split on top-level commas."""
    text = re.sub(r"//.*|/\*.*?\*/", "", text, flags=re.S)
    out = []
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9]*)(?:_([a-z]+))?\s*\{(.*?)\}", text, flags=re.S):
        entries, depth, cur = [], 0, ""
        for ch in m.group(3):
            depth += ch in "([" and 1 or ch in ")]" and -1 or 0
            if ch == "," and depth == 0:
                entries.append(cur)
                cur = ""
            else:
                cur += ch
        if cur.strip():
            entries.append(cur)
        out.append((m.group(1), m.group(2) or "", [e.strip() for e in entries if e.strip()]))
    return out


def parse_entry(e):
    m = ENTRY.match(e)
    if not m:
        raise SyntaxError(f"bad entry {e!r}")
    r0, r0b, r1, c0, c0b, label, expr = m.groups()
    return dict(r0=r0, r0b=r0b, r1=r1, c0=c0, c0b=c0b, label=label, expr=expr)


def translate(text):
    bl = blocks(text)
    inputs = []
    jl = ["# generated by dfn/ds2jl.py from dfn/dfn.ds; do not edit", "using SparseArrays, LinearAlgebra", ""]
    body = []  # statements inside dfn_eval
    state = []  # (name, start, stop)
    mass = []  # (start, stop, value)
    F = out = u0 = None
    deps = {}  # varying/event tensor -> identifiers it uses

    def ranged(entries, kind):
        """vector block -> Julia expression (vcat of pieces, ranges broadcast to their length)"""
        parts = []
        for e in map(parse_entry, entries):
            if e["c0"] is not None:
                raise SyntaxError(f"unexpected matrix entry in {kind}")
            ex = emit(Parser(tokenize(e["expr"])).expr(), inputs)
            if e["r0"] is not None:
                if e["r0b"] is not None:
                    raise SyntaxError("'..' range in a vector")
                a, b = int(e["r0"]), int(e["r1"]) if e["r1"] else int(e["r0"]) + 1
                parts.append(f"bc({ex}, {b - a})")
            else:
                parts.append(ex)
        return parts[0] if len(parts) == 1 else "vcat(" + ", ".join(parts) + ")"

    for name, idx, entries in bl:
        if name == "in":
            for e in map(parse_entry, entries):
                inputs.append(e["label"])
        elif name.startswith("constant") and idx == "ij":
            I, J, V = [], [], []
            for e in map(parse_entry, entries):
                v = repr(float(e["expr"]))
                r0, c0 = int(e["r0"]), int(e["c0"])
                if e["r0b"]:  # diagonal (a..b, c..d)
                    for k in range(int(e["r0b"]) - r0):
                        I.append(r0 + k + 1), J.append(c0 + k + 1), V.append(v)
                else:
                    I.append(r0 + 1), J.append(c0 + 1), V.append(v)
            jl.append(f"const {name} = sparse({I}, {J}, [{', '.join(V)}], {max(I)}, {max(J)})")
        elif name.startswith("constant"):
            vals = []
            for e in map(parse_entry, entries):
                n = int(e["r1"]) - int(e["r0"]) if e["r0"] else 1
                vals += [repr(float(e["expr"]))] * n
            jl.append(f"const {name} = [{', '.join(vals)}]")
        elif name == "u":
            u0 = []
            for e in map(parse_entry, entries):
                u0.append(emit(Parser(tokenize(e["expr"])).expr(), inputs))
                state.append(e["label"])
        elif name == "dudt":
            pos, spans = 0, []
            for e in map(parse_entry, entries):
                a, b = (int(e["r0"]), int(e["r1"])) if e["r0"] else (pos, pos + 1)
                spans.append((a, b))
                pos = b
            state = [(s, a, b) for s, (a, b) in zip(state, spans)]
            for s, a, b in state:
                body.append(f"{s} = @view u[{a + 1}:{b}]")
        elif name == "M":
            for e in map(parse_entry, entries):
                if e["r0"]:
                    a, b = int(e["r0"]), int(e["r1"])
                    if float(e["expr"]) != 0:
                        raise SyntaxError("non-zero constant mass row")
                    mass.append((a, b, 0.0))
                else:
                    s = next(x for x in state if f"d{x[0]}dt" == base(e["expr"]))
                    a, b = s[1], s[2]
                    mass.append((a, b, 1.0))
        elif name == "F":
            F = ranged(entries, "F")
        elif name == "out":
            out = ranged(entries, "out")
        elif name in ("stop", "reset"):
            continue
        elif name.startswith(("varying", "event")):
            body.append(f"{name} = {ranged(entries, name)}")
            deps[name] = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", body[-1].split("=", 1)[1]))
        else:
            raise SyntaxError(f"unsupported block {name}_{idx}")

    nstates = state[-1][2]
    # statements the output needs, so dfn_out (differentiated per saved time) skips the rest
    need, todo = set(), set(re.findall(r"[A-Za-z][A-Za-z0-9]*", out))
    while todo:
        n = todo.pop()
        if n in deps and n not in need:
            need.add(n)
            todo |= deps[n]
    out_body = [b for b in body if " = @view" in b or b.split(" = ")[0] in need]
    unpack = [f"{n} = p[{k + 1}]" for k, n in enumerate(inputs)]
    jl += [
        "",
        "bc(x::Number, n) = fill(x, n)",
        "bc(x::AbstractVector, n) = length(x) == n ? x : fill(x[1], n)",
        "",
        f"const DFN_NSTATES = {nstates}",
        f"const DFN_NPARAMS = {len(inputs)}",
        "",
        "function dfn_eval(u, p, t)",
        *["    " + s for s in unpack + body],
        f"    F = {F}",
        f"    out = {out}",
        "    F, out",
        "end",
        "",
        "function dfn_rhs!(du, u, p, t)",
        "    du .= dfn_eval(u, p, t)[1]",
        "    nothing",
        "end",
        "",
        "function dfn_out(u, p, t)",
        *["    " + s for s in unpack + out_body],
        f"    {out}",
        "end",
        "",
        "function dfn_u0(p)",
        *["    " + s for s in unpack],
        f"    vcat({', '.join(u0)})",
        "end",
        "",
        "const dfn_mass = let d = zeros(DFN_NSTATES)",
        *[f"    d[{a + 1}:{b}] .= {v!r}" for a, b, v in mass],
        "    Diagonal(d)",
        "end",
    ]
    return "\n".join(jl) + "\n"


if __name__ == "__main__":
    src, dst = sys.argv[1:3]
    with open(src) as f:
        code = translate(f.read())
    with open(dst, "w") as f:
        f.write(code)

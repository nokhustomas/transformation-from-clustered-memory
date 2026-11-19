# -*- coding: utf-8 -*-
"""
实验六语料构建: 从本地已安装 Python 包(torch/numpy/sklearn)的源码里用 ast 抽取
全部 docstring,拼成一个真实英文语料。

为什么用这个而不是下载公开语料: 本环境的网页抓取工具是"按提示回答页面内容"型,
不适合搬运 MB 级原文;包源码就在本地磁盘上,docstring 是人写的、语法完整的
英文技术散文,量足(数 MB)、确定性强、无网络依赖。局限如实记录: 领域偏技术文档,
词汇分布不等于通用英语——对"组会不会在真实语言上涌现"这个问题,它是真实语言,
但结论外推到通用语料时要注明领域。
"""

import ast
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "docstring_corpus.txt")
PKGS = ["torch", "numpy", "sklearn"]
MAX_BYTES = 4_000_000


def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("test", "tests", "testing", "__pycache__")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def extract_docstrings(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, ValueError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node)
            if d and len(d) > 80:  # 太短的多为占位
                out.append(d)
    return out


def clean(text):
    lines = []
    for ln in text.split("\n"):
        ln = ln.rstrip()
        # 过滤明显非散文的行(签名/表格分隔/doctest提示符等保留一部分真实感,但去掉纯符号行)
        stripped = ln.strip()
        if stripped and all(c in "-=~^#*+|:. " for c in stripped):
            continue
        lines.append(ln)
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    seen = set()
    total = 0
    n_docs = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for pkg in PKGS:
            try:
                mod = __import__(pkg)
            except ImportError:
                print(f"跳过未安装的 {pkg}")
                continue
            root = os.path.dirname(mod.__file__)
            for path in iter_py_files(root):
                for d in extract_docstrings(path):
                    key = hash(d)
                    if key in seen:
                        continue
                    seen.add(key)
                    c = clean(d)
                    b = (c + "\n\n").encode("utf-8")
                    f.write(c + "\n\n")
                    total += len(b)
                    n_docs += 1
                    if total >= MAX_BYTES:
                        break
                if total >= MAX_BYTES:
                    break
            print(f"{pkg}: 累计 {total/1e6:.2f}MB / {n_docs} 段", flush=True)
            if total >= MAX_BYTES:
                break
    print(f"完成: {OUT} ({total/1e6:.2f}MB, {n_docs} 段 docstring)")


if __name__ == "__main__":
    main()

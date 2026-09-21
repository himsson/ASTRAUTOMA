"""Парсер и сериализатор формата ConfigNode (KSP .cfg / .craft / .sfs).

Формат:
    KEY = value            // комментарий
    NODE
    {
        KEY = value
        CHILD { ... }
    }

Ключи повторяются (несколько RESOURCE, несколько link), поэтому значения
хранятся списком пар, а не словарём.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConfigNode:
    name: str = ""
    values: list[tuple[str, str]] = field(default_factory=list)
    comments: dict[str, str] = field(default_factory=dict)
    children: list["ConfigNode"] = field(default_factory=list)

    # ---------------- чтение ----------------
    def get(self, key: str, default: str | None = None) -> str | None:
        for k, v in self.values:
            if k == key:
                return v
        return default

    def get_all(self, key: str) -> list[str]:
        return [v for k, v in self.values if k == key]

    def get_float(self, key: str, default: float = 0.0) -> float:
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return float(raw.split()[0].rstrip(","))
        except (ValueError, IndexError):
            return default

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self.get_float(key, default))

    def get_vector(self, key: str) -> list[float]:
        raw = self.get(key)
        if raw is None:
            return []
        out = []
        for chunk in raw.replace(",", " ").split():
            try:
                out.append(float(chunk))
            except ValueError:
                break
        return out

    def comment(self, key: str) -> str:
        return self.comments.get(key, "")

    def nodes(self, name: str) -> list["ConfigNode"]:
        return [c for c in self.children if c.name == name]

    def node(self, name: str) -> "ConfigNode | None":
        for c in self.children:
            if c.name == name:
                return c
        return None

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    # ---------------- запись ----------------
    def set(self, key: str, value) -> "ConfigNode":
        self.values.append((key, str(value)))
        return self

    def add(self, child: "ConfigNode") -> "ConfigNode":
        self.children.append(child)
        return child

    def add_node(self, name: str) -> "ConfigNode":
        return self.add(ConfigNode(name=name))

    def render(self, indent: int = 0) -> str:
        pad = "\t" * indent
        lines = []
        if self.name:
            lines.append(f"{pad}{self.name}")
            lines.append(f"{pad}{{")
            inner = indent + 1
        else:
            inner = indent
        ipad = "\t" * inner
        for key, value in self.values:
            lines.append(f"{ipad}{key} = {value}")
        for child in self.children:
            lines.append(child.render(inner))
        if self.name:
            lines.append(f"{pad}}}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
def parse(text: str, name: str = "") -> ConfigNode:
    """Разбирает текст ConfigNode. Возвращает корневой узел."""
    root = ConfigNode(name=name)
    stack = [root]
    pending_name: str | None = None

    for raw_line in text.lstrip("﻿").splitlines():
        line = raw_line.strip().lstrip("﻿")
        if not line or line.startswith("//"):
            continue

        # Отделяем комментарий, сохраняя его (нужен для локализованных строк)
        comment = ""
        if "//" in line:
            line, comment = line.split("//", 1)
            line = line.strip()
            comment = comment.strip()
            if not line:
                continue

        while line:
            if line.startswith("{"):
                node = ConfigNode(name=pending_name or "")
                stack[-1].children.append(node)
                stack.append(node)
                pending_name = None
                line = line[1:].strip()
                continue
            if line.startswith("}"):
                if len(stack) > 1:
                    stack.pop()
                line = line[1:].strip()
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                # значение может закрывать узел: "key = value }"
                if value.endswith("}") and "{" not in value:
                    value = value[:-1].strip()
                    stack[-1].values.append((key, value))
                    if comment:
                        stack[-1].comments[key] = comment
                    if len(stack) > 1:
                        stack.pop()
                    line = ""
                    continue
                stack[-1].values.append((key, value))
                if comment:
                    stack[-1].comments[key] = comment
                line = ""
                continue
            # имя следующего узла
            token, _, rest = line.partition("{")
            pending_name = token.strip()
            line = ("{" + rest) if rest or "{" in raw_line else ""
    return root


def parse_file(path: str | Path) -> ConfigNode:
    # cfg-файлы KSP часто идут с BOM — utf-8-sig снимает его
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return parse(text, name="")


def localized(node: ConfigNode, key: str, default: str = "") -> str:
    """Достаёт человекочитаемое значение из #autoLOC-ключа.

    В cfg KSP пишет:  title = #autoLOC_500535 //#autoLOC_500535 = FL-T800 Fuel Tank
    """
    raw = node.get(key)
    if raw is None:
        return default
    if not raw.startswith("#"):
        return raw
    comment = node.comment(key)
    if "=" in comment:
        return comment.split("=", 1)[1].strip()
    return default or raw

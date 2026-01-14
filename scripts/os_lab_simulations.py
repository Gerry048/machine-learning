from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from html import escape


@dataclass
class Process:
    name: str
    arrival: int
    burst: int
    priority: int
    remaining: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.burst


def simulate_hpf(processes: List[Process], time_slice: int = 1) -> str:
    time = 0
    ready: List[Process] = []
    running: Process | None = None
    timeline: List[str] = []
    waiting_times = {p.name: 0 for p in processes}

    def enqueue_arrivals(t: int) -> None:
        for p in processes:
            if p.arrival == t:
                ready.append(p)

    def pick_next() -> Process | None:
        if not ready:
            return None
        ready.sort(key=lambda p: (-p.priority, p.arrival, p.name))
        return ready.pop(0)

    total_burst = sum(p.burst for p in processes)
    completed = 0

    enqueue_arrivals(time)

    while completed < total_burst:
        if running is None:
            running = pick_next()
        if running is None:
            timeline.append(f"t={time}: idle")
            time += time_slice
            enqueue_arrivals(time)
            for p in ready:
                p.priority += 1
                waiting_times[p.name] += time_slice
            continue

        running.remaining -= time_slice
        running.priority -= 2
        timeline.append(
            f"t={time}: run {running.name} (prio={running.priority + 2} -> {running.priority})"
        )

        time += time_slice
        enqueue_arrivals(time)

        for p in ready:
            p.priority += 1
            waiting_times[p.name] += time_slice

        if running.remaining > 0:
            ready.append(running)
        running = None
        completed += time_slice

    timeline.append("\nFinal priorities and waiting times:")
    for p in sorted(processes, key=lambda p: p.name):
        timeline.append(
            f"{p.name}: final priority={p.priority}, waiting={waiting_times[p.name]}"
        )
    return "\n".join(timeline)


def bankers_algorithm(
    allocation: List[Tuple[int, int, int]],
    max_need: List[Tuple[int, int, int]],
    total: Tuple[int, int, int],
) -> str:
    n = len(allocation)
    available = [
        total[i] - sum(allocation[j][i] for j in range(n)) for i in range(3)
    ]
    need = [
        [max_need[i][r] - allocation[i][r] for r in range(3)] for i in range(n)
    ]

    finish = [False] * n
    safe_sequence: List[int] = []

    def can_finish(i: int) -> bool:
        return all(need[i][r] <= available[r] for r in range(3))

    while len(safe_sequence) < n:
        progressed = False
        for i in range(n):
            if not finish[i] and can_finish(i):
                for r in range(3):
                    available[r] += allocation[i][r]
                finish[i] = True
                safe_sequence.append(i)
                progressed = True
        if not progressed:
            break

    lines = [
        f"Initial Available: {available}",
        "Need Matrix:",
    ]
    for i in range(n):
        lines.append(f"P{i}: {need[i]}")

    if len(safe_sequence) == n:
        seq = " -> ".join(f"P{i}" for i in safe_sequence)
        lines.append(f"Safe sequence found: {seq}")
    else:
        lines.append("No safe sequence found.")

    return "\n".join(lines)


@dataclass
class MemoryBlock:
    start: int
    size: int

    def __str__(self) -> str:
        return f"[{self.start}, {self.start + self.size}) size={self.size}"


def best_fit_allocation(total_size: int, blocks: List[int], processes: List[int]) -> str:
    free_blocks = [MemoryBlock(start, size) for start, size in blocks]
    allocations: List[Tuple[int, MemoryBlock]] = []

    def allocate(p_size: int) -> MemoryBlock | None:
        candidates = [b for b in free_blocks if b.size >= p_size]
        if not candidates:
            return None
        best = min(candidates, key=lambda b: b.size)
        free_blocks.remove(best)
        allocated_block = MemoryBlock(best.start, p_size)
        remaining = best.size - p_size
        if remaining > 0:
            free_blocks.append(MemoryBlock(best.start + p_size, remaining))
        return allocated_block

    lines = ["Best-fit allocation process:"]
    for idx, p_size in enumerate(processes, start=1):
        block = allocate(p_size)
        if block is None:
            lines.append(f"P{idx} size={p_size}: allocation failed")
        else:
            allocations.append((idx, block))
            lines.append(f"P{idx} size={p_size}: allocated {block}")

    lines.append("\nFree blocks after allocation:")
    for b in sorted(free_blocks, key=lambda b: b.start):
        lines.append(str(b))

    lines.append("\nReclaiming processes:")
    for idx, block in allocations:
        free_blocks.append(block)
        lines.append(f"P{idx} released {block}")

    free_blocks = sorted(free_blocks, key=lambda b: b.start)
    lines.append("\nFree blocks after reclaim:")
    for b in free_blocks:
        lines.append(str(b))

    return "\n".join(lines)


def lru_page_replacement(pages: List[int], frame_count: int) -> str:
    frames: List[int] = []
    recent: List[int] = []
    hits = 0
    faults = 0
    lines = ["LRU Page Replacement:", f"Frames: {frame_count}"]

    for step, page in enumerate(pages, start=1):
        if page in frames:
            hits += 1
            recent.remove(page)
            recent.append(page)
            status = "hit"
        else:
            faults += 1
            if len(frames) < frame_count:
                frames.append(page)
            else:
                lru_page = recent.pop(0)
                idx = frames.index(lru_page)
                frames[idx] = page
            recent.append(page)
            status = "fault"
        lines.append(
            f"{step:02d}: page={page} -> {status}, frames={frames}"
        )

    lines.append(f"\nTotal hits: {hits}")
    lines.append(f"Total faults: {faults}")
    return "\n".join(lines)


def render_text_image(text: str, output_path: str) -> None:
    lines = text.splitlines() or [""]
    padding = 12
    font_size = 14
    line_height = font_size + 6
    max_len = max(len(line) for line in lines)
    char_width = font_size * 0.6
    width = int(max_len * char_width) + padding * 2
    height = line_height * len(lines) + padding * 2
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<g font-family="monospace" font-size="{font_size}" fill="black">',
    ]
    y = padding + font_size
    for line in lines:
        svg_lines.append(f'<text x="{padding}" y="{y}">{escape(line)}</text>')
        y += line_height
    svg_lines.append("</g></svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def main() -> None:
    processes = [
        Process("P1", arrival=0, burst=3, priority=5),
        Process("P2", arrival=1, burst=4, priority=4),
        Process("P3", arrival=2, burst=2, priority=6),
        Process("P4", arrival=3, burst=3, priority=3),
    ]
    hpf_output = simulate_hpf(processes)

    allocation = [(2, 1, 1), (1, 0, 2), (2, 1, 1), (1, 1, 0)]
    max_need = [(5, 3, 2), (2, 2, 3), (4, 2, 2), (1, 2, 1)]
    total = (9, 5, 6)
    banker_output = bankers_algorithm(allocation, max_need, total)

    blocks = [(0, 100), (100, 500), (600, 200), (800, 300), (1100, 600)]
    processes_sizes = [212, 417, 112]
    best_fit_output = best_fit_allocation(1700, blocks, processes_sizes)

    pages = [3, 5, 2, 1, 4, 2, 5, 3, 1, 2, 6, 4, 3, 2, 5, 7, 2, 4, 1, 6]
    lru_output = lru_page_replacement(pages, frame_count=3)

    outputs = {
        "hpf": hpf_output,
        "banker": banker_output,
        "best_fit": best_fit_output,
        "lru": lru_output,
    }

    for key, text in outputs.items():
        render_text_image(text, f"docs/assets/{key}_output.svg")


if __name__ == "__main__":
    main()

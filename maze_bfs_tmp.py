from collections import deque

# Grid definition: True = allowed, False = forbidden (B)
# S=start, E=end, .=allowed, B=forbidden
# 1-indexed

grid_raw = [
    "SBBBBBBBB",  # row 1
    ".BBBBBBBB",  # row 2
    ".BBB....B",  # row 3
    "...B.BB.B",  # row 4
    "BB.B.BB.B",  # row 5
    "B..B.BB.B",  # row 6
    "B.BB.BB.B",  # row 7
    "B.BB.BB.B",  # row 8
    "B.BB.B..B",  # row 9
    "B....B.BB",  # row 10
    "BBBBBB.BB",  # row 11
    "BBBBBB.BB",  # row 12
    "BBB....BB",  # row 13
    "B...BBBBB",  # row 14
    "B.BBB...B",  # row 15
    "B..BB.B.B",  # row 16
    "BB.BB.B.B",  # row 17
    "BB..B.B.B",  # row 18
    "BBB...B.B",  # row 19
    "BBBBBBB.E",  # row 20
]

ROWS = 20
COLS = 9

def is_allowed(ch):
    return ch in ("S", ".", "E")

allowed = {}
for r in range(ROWS):
    for c in range(COLS):
        ch = grid_raw[r][c]
        allowed[(r+1, c+1)] = is_allowed(ch)

colors = {
    (1,1):"THEME", (1,2):"FF0099FF", (1,3):"FF0099FF", (1,4):"FF0099FF", (1,5):"FF0099FF", (1,6):"FF0099FF", (1,7):"FF0099FF", (1,8):"FF0099FF", (1,9):"FF0099FF",
    (2,1):"FF92D050", (2,2):"FF0099FF", (2,3):"FF0099FF", (2,4):"FF0099FF", (2,5):"FF0099FF", (2,6):"FF0099FF", (2,7):"FF0099FF", (2,8):"FF0099FF", (2,9):"FF0099FF",
    (3,1):"FFF478A7", (3,2):"FF0099FF", (3,3):"FF0099FF", (3,4):"FF0099FF", (3,5):"FFF478A7", (3,6):"FFFFFF00", (3,7):"FF92D050", (3,8):"FF92D050", (3,9):"FF0099FF",
    (4,1):"FFFFFF00", (4,2):"FFFFFF00", (4,3):"FF92D050", (4,4):"FF0099FF", (4,5):"FF92D050", (4,6):"FF0099FF", (4,7):"FF0099FF", (4,8):"FFFFFF00", (4,9):"FF0099FF",
    (5,1):"FF0099FF", (5,2):"FF0099FF", (5,3):"FF92D050", (5,4):"FF0099FF", (5,5):"FFFFFF00", (5,6):"FF0099FF", (5,7):"FF0099FF", (5,8):"FFFFFF00", (5,9):"FF0099FF",
    (6,1):"FF0099FF", (6,2):"FF92D050", (6,3):"FFFFFF00", (6,4):"FF0099FF", (6,5):"FF92D050", (6,6):"FF0099FF", (6,7):"FF0099FF", (6,8):"FF92D050", (6,9):"FF0099FF",
    (7,1):"FF0099FF", (7,2):"FFF478A7", (7,3):"FF0099FF", (7,4):"FF0099FF", (7,5):"FFFFFF00", (7,6):"FF0099FF", (7,7):"FF0099FF", (7,8):"FFFFFF00", (7,9):"FF0099FF",
    (8,1):"FF0099FF", (8,2):"FFFFFF00", (8,3):"FF0099FF", (8,4):"FF0099FF", (8,5):"FFFFFF00", (8,6):"FF0099FF", (8,7):"FF0099FF", (8,8):"FF92D050", (8,9):"FF0099FF",
    (9,1):"FF0099FF", (9,2):"FFFFFF00", (9,3):"FF0099FF", (9,4):"FF0099FF", (9,5):"FF92D050", (9,6):"FF0099FF", (9,7):"FFF478A7", (9,8):"FFF478A7", (9,9):"FF0099FF",
    (10,1):"FF0099FF", (10,2):"FF92D050", (10,3):"FF92D050", (10,4):"FFFFFF00", (10,5):"FFF478A7", (10,6):"FF0099FF", (10,7):"FF92D050", (10,8):"FF0099FF", (10,9):"FF0099FF",
    (11,1):"FF0099FF", (11,2):"FF0099FF", (11,3):"FF0099FF", (11,4):"FF0099FF", (11,5):"FF0099FF", (11,6):"FF0099FF", (11,7):"FFFFFF00", (11,8):"FF0099FF", (11,9):"FF0099FF",
    (12,1):"FF0099FF", (12,2):"FF0099FF", (12,3):"FF0099FF", (12,4):"FF0099FF", (12,5):"FF0099FF", (12,6):"FF0099FF", (12,7):"FFFFFF00", (12,8):"FF0099FF", (12,9):"FF0099FF",
    (13,1):"FF0099FF", (13,2):"FF0099FF", (13,3):"FF0099FF", (13,4):"FF92D050", (13,5):"FF92D050", (13,6):"FF92D050", (13,7):"FF92D050", (13,8):"FF0099FF", (13,9):"FF0099FF",
    (14,1):"FF0099FF", (14,2):"FFF478A7", (14,3):"FF92D050", (14,4):"FFFFFF00", (14,5):"FF0099FF", (14,6):"FF0099FF", (14,7):"FF0099FF", (14,8):"FF0099FF", (14,9):"FF0099FF",
    (15,1):"FF0099FF", (15,2):"FFF478A7", (15,3):"FF0099FF", (15,4):"FF0099FF", (15,5):"FF0099FF", (15,6):"FFFFFF00", (15,7):"FF92D050", (15,8):"FF92D050", (15,9):"FF0099FF",
    (16,1):"FF0099FF", (16,2):"FFFFFF00", (16,3):"FFFFFF00", (16,4):"FF0099FF", (16,5):"FF0099FF", (16,6):"FF92D050", (16,7):"FF0099FF", (16,8):"FFF478A7", (16,9):"FF0099FF",
    (17,1):"FF0099FF", (17,2):"FF0099FF", (17,3):"FF92D050", (17,4):"FF0099FF", (17,5):"FF0099FF", (17,6):"FFFFFF00", (17,7):"FF0099FF", (17,8):"FF92D050", (17,9):"FF0099FF",
    (18,1):"FF0099FF", (18,2):"FF0099FF", (18,3):"FF92D050", (18,4):"FFFFFF00", (18,5):"FF0099FF", (18,6):"FFF478A7", (18,7):"FF0099FF", (18,8):"FFFFFF00", (18,9):"FF0099FF",
    (19,1):"FF0099FF", (19,2):"FF0099FF", (19,3):"FF0099FF", (19,4):"FFF478A7", (19,5):"FF92D050", (19,6):"FFF478A7", (19,7):"FF0099FF", (19,8):"FFF478A7", (19,9):"FF0099FF",
    (20,1):"FF0099FF", (20,2):"FF0099FF", (20,3):"FF0099FF", (20,4):"FF0099FF", (20,5):"FF0099FF", (20,6):"FF0099FF", (20,7):"FF0099FF", (20,8):"FF92D050", (20,9):"THEME_END",
}

DIRECTIONS = {
    "U": (-2, 0),
    "D": (2, 0),
    "L": (0, -2),
    "R": (0, 2),
}

OPPOSITES = {"U": "D", "D": "U", "L": "R", "R": "L"}

END = (20, 9)

def col_to_letter(c):
    return chr(ord("A") + c - 1)

def format_cell(r, c):
    return f"{col_to_letter(c)}{r}"

def dfs_all_solutions():
    solutions = []
    # Stack: (row, col, last_dir, path, visited_states)
    stack = [(1, 1, None, [(1, 1)], frozenset({(1, 1, None)}))]

    while stack:
        r, c, last_dir, path, visited = stack.pop()

        if (r, c) == END:
            solutions.append(path)
            continue

        for d, (dr, dc) in DIRECTIONS.items():
            if last_dir is not None and d == OPPOSITES[last_dir]:
                continue

            nr, nc = r + dr, c + dc

            if not (1 <= nr <= ROWS and 1 <= nc <= COLS):
                continue

            if not allowed.get((nr, nc), False):
                continue

            new_state = (nr, nc, d)
            if new_state in visited:
                continue

            stack.append((nr, nc, d, path + [(nr, nc)], visited | {new_state}))

    return solutions

print("Running DFS to find all solutions...")
solutions = dfs_all_solutions()

# Sort by length (shortest first)
solutions.sort(key=lambda p: len(p))

print(f"\nNumber of solutions: {len(solutions)}")

if solutions:
    first = solutions[0]
    print(f"\nFirst solution (shortest, {len(first)-1} moves, {len(first)} positions including start):")
    for i, (r, c) in enumerate(first):
        cell = format_cell(r, c)
        color = colors.get((r, c), "UNKNOWN")
        print(f"  Turn {i}: ({r},{c}) = {cell}, color={color}")

    print(f"\nCell at turn 11 (index 11):")
    if len(first) > 11:
        r11, c11 = first[11]
        cell11 = format_cell(r11, c11)
        color11 = colors.get((r11, c11), "UNKNOWN")
        last6 = color11[-6:] if len(color11) >= 6 else color11
        print(f"  Position: ({r11},{c11}) = {cell11}")
        print(f"  Full color code: {color11}")
        print(f"  Last 6 characters: {last6}")
    else:
        print("  Path has fewer than 12 positions!")

    print(f"\nAll solutions lengths (number of positions): {[len(s) for s in solutions]}")

from collections import deque

grid_raw = [
    "S B B B B B B B B",
    ". B B B B B B B B",
    ". B B B . . . . B",
    ". . . B . B B . B",
    "B B . B . B B . B",
    "B . . B . B B . B",
    "B . B B . B B . B",
    "B . B B . B B . B",
    "B . B B . B . . B",
    "B . . . . B . B B",
    "B B B B B B . B B",
    "B B B B B B . B B",
    "B B B . . . . B B",
    "B . . . B B B B B",
    "B . B B B . . . B",
    "B . . B B . B . B",
    "B B . B B . B . B",
    "B B . . B . B . B",
    "B B B . . . B . B",
    "B B B B B B B . E",
]

grid = {}
for r_idx, row_str in enumerate(grid_raw):
    cells = row_str.split()
    for c_idx, cell in enumerate(cells):
        row = r_idx + 1
        col = c_idx + 1
        grid[(row, col)] = cell

def is_allowed(row, col):
    cell = grid.get((row, col))
    return cell in ("S", "E", ".")

colors = {
    (1,1): "THEME", (1,2): "FF0099FF", (1,3): "FF0099FF", (1,4): "FF0099FF", (1,5): "FF0099FF", (1,6): "FF0099FF", (1,7): "FF0099FF", (1,8): "FF0099FF", (1,9): "FF0099FF",
    (2,1): "FF92D050", (2,2): "FF0099FF", (2,3): "FF0099FF", (2,4): "FF0099FF", (2,5): "FF0099FF", (2,6): "FF0099FF", (2,7): "FF0099FF", (2,8): "FF0099FF", (2,9): "FF0099FF",
    (3,1): "FFF478A7", (3,2): "FF0099FF", (3,3): "FF0099FF", (3,4): "FF0099FF", (3,5): "FFF478A7", (3,6): "FFFFFF00", (3,7): "FF92D050", (3,8): "FF92D050", (3,9): "FF0099FF",
    (4,1): "FFFFFF00", (4,2): "FFFFFF00", (4,3): "FF92D050", (4,4): "FF0099FF", (4,5): "FF92D050", (4,6): "FF0099FF", (4,7): "FF0099FF", (4,8): "FFFFFF00", (4,9): "FF0099FF",
    (5,1): "FF0099FF", (5,2): "FF0099FF", (5,3): "FF92D050", (5,4): "FF0099FF", (5,5): "FFFFFF00", (5,6): "FF0099FF", (5,7): "FF0099FF", (5,8): "FFFFFF00", (5,9): "FF0099FF",
    (6,1): "FF0099FF", (6,2): "FF92D050", (6,3): "FFFFFF00", (6,4): "FF0099FF", (6,5): "FF92D050", (6,6): "FF0099FF", (6,7): "FF0099FF", (6,8): "FF92D050", (6,9): "FF0099FF",
    (7,1): "FF0099FF", (7,2): "FFF478A7", (7,3): "FF0099FF", (7,4): "FF0099FF", (7,5): "FFFFFF00", (7,6): "FF0099FF", (7,7): "FF0099FF", (7,8): "FFFFFF00", (7,9): "FF0099FF",
    (8,1): "FF0099FF", (8,2): "FFFFFF00", (8,3): "FF0099FF", (8,4): "FF0099FF", (8,5): "FFFFFF00", (8,6): "FF0099FF", (8,7): "FF0099FF", (8,8): "FF92D050", (8,9): "FF0099FF",
    (9,1): "FF0099FF", (9,2): "FFFFFF00", (9,3): "FF0099FF", (9,4): "FF0099FF", (9,5): "FF92D050", (9,6): "FF0099FF", (9,7): "FFF478A7", (9,8): "FFF478A7", (9,9): "FF0099FF",
    (10,1): "FF0099FF", (10,2): "FF92D050", (10,3): "FF92D050", (10,4): "FFFFFF00", (10,5): "FFF478A7", (10,6): "FF0099FF", (10,7): "FF92D050", (10,8): "FF0099FF", (10,9): "FF0099FF",
    (11,1): "FF0099FF", (11,2): "FF0099FF", (11,3): "FF0099FF", (11,4): "FF0099FF", (11,5): "FF0099FF", (11,6): "FF0099FF", (11,7): "FFFFFF00", (11,8): "FF0099FF", (11,9): "FF0099FF",
    (12,1): "FF0099FF", (12,2): "FF0099FF", (12,3): "FF0099FF", (12,4): "FF0099FF", (12,5): "FF0099FF", (12,6): "FF0099FF", (12,7): "FFFFFF00", (12,8): "FF0099FF", (12,9): "FF0099FF",
    (13,1): "FF0099FF", (13,2): "FF0099FF", (13,3): "FF0099FF", (13,4): "FF92D050", (13,5): "FF92D050", (13,6): "FF92D050", (13,7): "FF92D050", (13,8): "FF0099FF", (13,9): "FF0099FF",
    (14,1): "FF0099FF", (14,2): "FFF478A7", (14,3): "FF92D050", (14,4): "FFFFFF00", (14,5): "FF0099FF", (14,6): "FF0099FF", (14,7): "FF0099FF", (14,8): "FF0099FF", (14,9): "FF0099FF",
    (15,1): "FF0099FF", (15,2): "FFF478A7", (15,3): "FF0099FF", (15,4): "FF0099FF", (15,5): "FF0099FF", (15,6): "FFFFFF00", (15,7): "FF92D050", (15,8): "FF92D050", (15,9): "FF0099FF",
    (16,1): "FF0099FF", (16,2): "FFFFFF00", (16,3): "FFFFFF00", (16,4): "FF0099FF", (16,5): "FF0099FF", (16,6): "FF92D050", (16,7): "FF0099FF", (16,8): "FFF478A7", (16,9): "FF0099FF",
    (17,1): "FF0099FF", (17,2): "FF0099FF", (17,3): "FF92D050", (17,4): "FF0099FF", (17,5): "FF0099FF", (17,6): "FFFFFF00", (17,7): "FF0099FF", (17,8): "FF92D050", (17,9): "FF0099FF",
    (18,1): "FF0099FF", (18,2): "FF0099FF", (18,3): "FF92D050", (18,4): "FFFFFF00", (18,5): "FF0099FF", (18,6): "FFF478A7", (18,7): "FF0099FF", (18,8): "FFFFFF00", (18,9): "FF0099FF",
    (19,1): "FF0099FF", (19,2): "FF0099FF", (19,3): "FF0099FF", (19,4): "FFF478A7", (19,5): "FF92D050", (19,6): "FFF478A7", (19,7): "FF0099FF", (19,8): "FFF478A7", (19,9): "FF0099FF",
    (20,1): "FF0099FF", (20,2): "FF0099FF", (20,3): "FF0099FF", (20,4): "FF0099FF", (20,5): "FF0099FF", (20,6): "FF0099FF", (20,7): "FF0099FF", (20,8): "FF92D050", (20,9): "THEME_END",
}

col_letter = {1:"A", 2:"B", 3:"C", 4:"D", 5:"E", 6:"F", 7:"G", 8:"H", 9:"I"}

def cell_name(row, col):
    return col_letter[col] + str(row)

opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
dir_delta = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

START = (1, 1)
END = (20, 9)

all_solutions = []

def dfs(row, col, last_dir, path, visited):
    if (row, col) == END:
        all_solutions.append(list(path))
        return
    for direction, (dr, dc) in dir_delta.items():
        if last_dir is not None and direction == opposite[last_dir]:
            continue
        nr, nc = row + dr, col + dc
        if not (1 <= nr <= 20 and 1 <= nc <= 9):
            continue
        if not is_allowed(nr, nc):
            continue
        if (nr, nc) in visited:
            continue
        visited.add((nr, nc))
        path.append((nr, nc, direction))
        dfs(nr, nc, direction, path, visited)
        path.pop()
        visited.remove((nr, nc))

visited_init = set()
visited_init.add(START)
path_init = [(1, 1, None)]
dfs(1, 1, None, path_init, visited_init)

all_solutions.sort(key=lambda s: len(s))

print("Number of solutions: " + str(len(all_solutions)))
print()

if all_solutions:
    shortest = all_solutions[0]
    print("First solution (shortest), length=" + str(len(shortest)) + " steps:")
    for i, (r, c, d) in enumerate(shortest):
        cname = cell_name(r, c)
        color = colors.get((r, c), "UNKNOWN")
        print("Turn " + str(i) + ": (" + str(r) + "," + str(c) + ") = " + cname + ", color=" + color)
    print()
    turn11 = shortest[11]
    r11, c11, d11 = turn11
    cname11 = cell_name(r11, c11)
    color11 = colors.get((r11, c11), "UNKNOWN")
    last6 = color11[-6:]
    print("Cell at turn 11: row=" + str(r11) + ", col=" + str(c11) + ", cell name=" + cname11)
    print("Full color code: " + color11)
    print("Last 6 characters of color: " + last6)

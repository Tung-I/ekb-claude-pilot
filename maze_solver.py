# Maze solver using DFS

grid_data = [
    (1,1,"NONE"),(1,2,"BLUE"),(1,3,"BLUE"),(1,4,"BLUE"),(1,5,"BLUE"),(1,6,"BLUE"),(1,7,"BLUE"),(1,8,"BLUE"),(1,9,"BLUE"),
    (2,1,"GRN"),(2,2,"BLUE"),(2,3,"BLUE"),(2,4,"BLUE"),(2,5,"BLUE"),(2,6,"BLUE"),(2,7,"BLUE"),(2,8,"BLUE"),(2,9,"BLUE"),
    (3,1,"PNK"),(3,2,"BLUE"),(3,3,"BLUE"),(3,4,"BLUE"),(3,5,"PNK"),(3,6,"YLW"),(3,7,"GRN"),(3,8,"GRN"),(3,9,"BLUE"),
    (4,1,"YLW"),(4,2,"YLW"),(4,3,"GRN"),(4,4,"BLUE"),(4,5,"GRN"),(4,6,"BLUE"),(4,7,"BLUE"),(4,8,"YLW"),(4,9,"BLUE"),
    (5,1,"BLUE"),(5,2,"BLUE"),(5,3,"GRN"),(5,4,"BLUE"),(5,5,"YLW"),(5,6,"BLUE"),(5,7,"BLUE"),(5,8,"YLW"),(5,9,"BLUE"),
    (6,1,"BLUE"),(6,2,"GRN"),(6,3,"YLW"),(6,4,"BLUE"),(6,5,"GRN"),(6,6,"BLUE"),(6,7,"BLUE"),(6,8,"GRN"),(6,9,"BLUE"),
    (7,1,"BLUE"),(7,2,"PNK"),(7,3,"BLUE"),(7,4,"BLUE"),(7,5,"YLW"),(7,6,"BLUE"),(7,7,"BLUE"),(7,8,"YLW"),(7,9,"BLUE"),
    (8,1,"BLUE"),(8,2,"YLW"),(8,3,"BLUE"),(8,4,"BLUE"),(8,5,"YLW"),(8,6,"BLUE"),(8,7,"BLUE"),(8,8,"GRN"),(8,9,"BLUE"),
    (9,1,"BLUE"),(9,2,"YLW"),(9,3,"BLUE"),(9,4,"BLUE"),(9,5,"GRN"),(9,6,"BLUE"),(9,7,"PNK"),(9,8,"PNK"),(9,9,"BLUE"),
    (10,1,"BLUE"),(10,2,"GRN"),(10,3,"GRN"),(10,4,"YLW"),(10,5,"PNK"),(10,6,"BLUE"),(10,7,"GRN"),(10,8,"BLUE"),(10,9,"BLUE"),
    (11,1,"BLUE"),(11,2,"BLUE"),(11,3,"BLUE"),(11,4,"BLUE"),(11,5,"BLUE"),(11,6,"BLUE"),(11,7,"YLW"),(11,8,"BLUE"),(11,9,"BLUE"),
    (12,1,"BLUE"),(12,2,"BLUE"),(12,3,"BLUE"),(12,4,"BLUE"),(12,5,"BLUE"),(12,6,"BLUE"),(12,7,"YLW"),(12,8,"BLUE"),(12,9,"BLUE"),
    (13,1,"BLUE"),(13,2,"BLUE"),(13,3,"BLUE"),(13,4,"GRN"),(13,5,"GRN"),(13,6,"GRN"),(13,7,"GRN"),(13,8,"BLUE"),(13,9,"BLUE"),
    (14,1,"BLUE"),(14,2,"PNK"),(14,3,"GRN"),(14,4,"YLW"),(14,5,"BLUE"),(14,6,"BLUE"),(14,7,"BLUE"),(14,8,"BLUE"),(14,9,"BLUE"),
    (15,1,"BLUE"),(15,2,"PNK"),(15,3,"BLUE"),(15,4,"BLUE"),(15,5,"BLUE"),(15,6,"YLW"),(15,7,"GRN"),(15,8,"GRN"),(15,9,"BLUE"),
    (16,1,"BLUE"),(16,2,"YLW"),(16,3,"YLW"),(16,4,"BLUE"),(16,5,"BLUE"),(16,6,"GRN"),(16,7,"BLUE"),(16,8,"PNK"),(16,9,"BLUE"),
    (17,1,"BLUE"),(17,2,"BLUE"),(17,3,"GRN"),(17,4,"BLUE"),(17,5,"BLUE"),(17,6,"YLW"),(17,7,"BLUE"),(17,8,"GRN"),(17,9,"BLUE"),
    (18,1,"BLUE"),(18,2,"BLUE"),(18,3,"GRN"),(18,4,"YLW"),(18,5,"BLUE"),(18,6,"PNK"),(18,7,"BLUE"),(18,8,"YLW"),(18,9,"BLUE"),
    (19,1,"BLUE"),(19,2,"BLUE"),(19,3,"BLUE"),(19,4,"PNK"),(19,5,"GRN"),(19,6,"PNK"),(19,7,"BLUE"),(19,8,"PNK"),(19,9,"BLUE"),
    (20,1,"BLUE"),(20,2,"BLUE"),(20,3,"BLUE"),(20,4,"BLUE"),(20,5,"BLUE"),(20,6,"BLUE"),(20,7,"BLUE"),(20,8,"GRN"),(20,9,"NONE"),
]

grid = {}
for (r, c, color) in grid_data:
    grid[(r, c)] = color

ROWS = 20
COLS = 9
START = (1, 1)
END = (20, 9)

COLOR_HEX = {
    "GRN": "92D050",
    "YLW": "FFFF00",
    "PNK": "F478A7",
    "BLUE": "0099FF",
    "NONE": None,
}

def is_valid(r, c):
    return 1 <= r <= ROWS and 1 <= c <= COLS and grid.get((r, c), "BLUE") != "BLUE"

def dfs():
    paths = []
    path_count = [0]
    MAX_PATHS = 10

    def recurse(r, c, path, visited):
        if path_count[0] >= MAX_PATHS:
            return
        if (r, c) == END:
            path_count[0] += 1
            paths.append(list(path))
            return
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and is_valid(nr, nc):
                visited.add((nr, nc))
                path.append((nr, nc))
                recurse(nr, nc, path, visited)
                path.pop()
                visited.remove((nr, nc))
                if path_count[0] >= MAX_PATHS:
                    return

    visited = {START}
    recurse(START[0], START[1], [START], visited)
    return paths

paths = dfs()

print("Total paths found (up to 10): " + str(len(paths)))
print()

for i, path in enumerate(paths):
    print("--- Path " + str(i+1) + " (length " + str(len(path)) + " cells) ---")
    print("Cells: " + " -> ".join("(" + str(r) + "," + str(c) + ")" for r, c in path))
    if len(path) > 11:
        cell_at_11 = path[11]
        color = grid.get(cell_at_11, "BLUE")
        hex_code = COLOR_HEX.get(color)
        if hex_code:
            print("  Cell at turn 11 (index 11): " + str(cell_at_11) + ", color=" + color + ", hex=" + hex_code)
        else:
            print("  Cell at turn 11 (index 11): " + str(cell_at_11) + ", color=" + color + ", hex=no color")
    else:
        print("  Path too short to have index 11 (length=" + str(len(path)) + ")")
    print()

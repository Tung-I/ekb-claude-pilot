
import re, zipfile
from collections import deque

xlsx_path = '/work/pi_rsitaram_umass_edu/tungi/ekb-claude-pilot/data/gaia/raw/65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx'
with zipfile.ZipFile(xlsx_path) as z:
    sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')

grid_styles = {}
cell_pat = re.compile(r'<c r="([A-Z]+)(\d+)" s="(\d+)"')
for m in cell_pat.finditer(sheet):
    col_str, row_str, s = m.group(1), m.group(2), int(m.group(3))
    col = ord(col_str) - ord("A")
    row = int(row_str) - 1
    grid_styles[(row, col)] = s

ROWS, COLS = 20, 9
for r in range(ROWS):
    for c in range(COLS):
        if (r,c) not in grid_styles:
            grid_styles[(r,c)] = 0

BLUE_STYLE = 5
def is_blue(r, c):
    return grid_styles.get((r,c), 0) == BLUE_STYLE

style_rgb = {
    0: "FFFFFF", 1: "FFFFFF", 2: "92D050",
    3: "FFFF00", 4: "F478A7", 5: "0099FF",
}

start = (0, 0)
end = (19, 8)
DIRS = [(-1,0,"UP"), (1,0,"DOWN"), (0,-1,"LEFT"), (0,1,"RIGHT")]
OPPOSITE = {"UP":"DOWN", "DOWN":"UP", "LEFT":"RIGHT", "RIGHT":"LEFT"}

# 1-step BFS (no backward, no revisit)
def bfs_1step(no_backward):
    queue = deque()
    queue.append((start[0], start[1], None, [start]))
    seen = set()
    solutions = []
    while queue:
        r, c, last_dir, path = queue.popleft()
        if len(path) > 30:
            continue
        key = (r, c, last_dir, frozenset(path))
        if key in seen:
            continue
        seen.add(key)
        path_set = set(path)
        for dr, dc, dname in DIRS:
            if no_backward and last_dir and dname == OPPOSITE[last_dir]:
                continue
            nr, nc = r+dr, c+dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS): continue
            if is_blue(nr, nc): continue
            if (nr, nc) in path_set: continue
            new_path = path + [(nr, nc)]
            if (nr, nc) == end:
                solutions.append(new_path)
                cells = [f"{chr(65+cc)}{rr+1}" for rr,cc in new_path]
                print(f"1-STEP SOLUTION ({len(new_path)-1} turns, no_backward={no_backward}): {cells}")
                if len(solutions) >= 3:
                    return solutions
            else:
                queue.append((nr, nc, dname, new_path))
    return solutions

print("=== 1-step movement, no backward ===")
sols = bfs_1step(True)
if not sols:
    print("\n=== 1-step movement, allow backward ===")
    sols = bfs_1step(False)

if sols:
    path = sols[0]
    print("\nPath details:")
    for j, (r, c) in enumerate(path):
        s = grid_styles[(r,c)]
        clr = style_rgb.get(s, "??????")
        print(f"  Turn {j}: {chr(65+c)}{r+1} style={s} color=FF{clr}")
    if len(path) >= 12:
        r11, c11 = path[11]
        s = grid_styles[(r11,c11)]
        clr = style_rgb.get(s, "??????")
        print(f"\nTURN 11 ANSWER (1-step): {chr(65+c11)}{r11+1}, color={clr}")
    else:
        print(f"Path only {len(path)-1} turns long!")
else:
    print("NO SOLUTION")

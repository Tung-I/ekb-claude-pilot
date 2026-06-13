
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

# Style to actual 6-digit hex (without alpha)
style_rgb = {
    0: "FFFFFF",
    1: "FFFFFF",
    2: "92D050",
    3: "FFFF00",
    4: "F478A7",
    5: "0099FF",
}

start = (0, 0)
end = (19, 8)

# Knight moves: (+-1, +-2) and (+-2, +-1)
KNIGHT_MOVES = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]

non_blue = [(r,c) for r in range(ROWS) for c in range(COLS) if not is_blue(r,c)]

# Check knight connectivity
graph_knight = {}
for r, c in non_blue:
    nbrs = []
    for dr, dc in KNIGHT_MOVES:
        nr, nc = r+dr, c+dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and not is_blue(nr, nc):
            nbrs.append((nr, nc))
    graph_knight[(r,c)] = nbrs

from collections import deque
def get_comp(start_cell, graph):
    comp = set()
    q = deque([start_cell])
    while q:
        cur = q.popleft()
        if cur in comp: continue
        comp.add(cur)
        for nb in graph.get(cur, []):
            if nb not in comp:
                q.append(nb)
    return comp

comp_start = get_comp(start, graph_knight)
print(f"Knight moves: START component size: {len(comp_start)}, END in it: {end in comp_start}")

if end in comp_start:
    # BFS to find path
    OPPOSITE_KNIGHT = None  # no "backward" concept for knight
    queue = deque()
    queue.append((start[0], start[1], None, [start]))
    seen = set()
    solutions = []
    while queue:
        r, c, last_move, path = queue.popleft()
        if len(path) > 30:
            continue
        key = (r, c, last_move, frozenset(path))
        if key in seen:
            continue
        seen.add(key)
        path_set = set(path)
        for dr, dc in KNIGHT_MOVES:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS): continue
            if is_blue(nr, nc): continue
            if (nr, nc) in path_set: continue
            new_path = path + [(nr, nc)]
            if (nr, nc) == end:
                solutions.append(new_path)
                cells = [f"{chr(65+cc)}{rr+1}" for rr,cc in new_path]
                print(f"SOLUTION ({len(new_path)-1} turns): {cells}")
                if len(solutions) >= 3:
                    break
            else:
                queue.append((nr, nc, (dr, dc), new_path))
        if len(solutions) >= 3:
            break

    if solutions:
        path = solutions[0]
        print("\nPath details:")
        for j, (r, c) in enumerate(path):
            s = grid_styles[(r,c)]
            clr = style_rgb.get(s, "??????")
            print(f"  Turn {j}: {chr(65+c)}{r+1} style={s} color={clr}")
        if len(path) >= 12:
            r11, c11 = path[11]
            s = grid_styles[(r11,c11)]
            clr = style_rgb.get(s, "??????")
            print(f"\nTURN 11 ANSWER (knight): {chr(65+c11)}{r11+1}, color={clr}")
    else:
        print("Knight: no solution found")
else:
    print("Knight: END not reachable from START")

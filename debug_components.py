
# Build graph from XML data directly
# Style mapping: s=1 -> white (passable), s=2 -> green, s=3 -> yellow, s=4 -> pink, s=5 -> BLUE (blocked)

# From sheet1.xml parsing:
grid_styles = {}  # (row, col) -> style_id (1-indexed row/col)

import re
import zipfile

xlsx_path = '/work/pi_rsitaram_umass_edu/tungi/ekb-claude-pilot/data/gaia/raw/65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx'
with zipfile.ZipFile(xlsx_path) as z:
    sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')

# Parse cells: <c r="A1" s="1" ...> or <c r="B3" s="5"/>
cell_pat = re.compile(r'<c r="([A-Z]+)(\d+)" s="(\d+)"')
for m in cell_pat.finditer(sheet):
    col_str, row_str, s = m.group(1), m.group(2), int(m.group(3))
    col = ord(col_str) - ord("A")
    row = int(row_str) - 1
    grid_styles[(row, col)] = s

ROWS, COLS = 20, 9
# Fill missing cells with default style 0 (passable)
for r in range(ROWS):
    for c in range(COLS):
        if (r,c) not in grid_styles:
            grid_styles[(r,c)] = 0

BLUE_STYLE = 5
def is_blue(r, c):
    return grid_styles.get((r,c), 0) == BLUE_STYLE

start = (0, 0)
end = (19, 8)

print("Grid styles:")
for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        s = grid_styles.get((r,c), 0)
        if (r,c) == start: row_str += "S"
        elif (r,c) == end: row_str += "E"
        elif s == BLUE_STYLE: row_str += "X"
        else: row_str += str(s)
    print(f"  Row {r+1:2d}: {row_str}")

# Build connectivity graph
DIRS = [(-1,0,"UP"), (1,0,"DOWN"), (0,-1,"LEFT"), (0,1,"RIGHT")]
graph = {}
for r in range(ROWS):
    for c in range(COLS):
        if is_blue(r, c):
            continue
        moves = []
        for dr, dc, dname in DIRS:
            mid_r, mid_c = r+dr, c+dc
            new_r, new_c = r+2*dr, c+2*dc
            if not (0 <= mid_r < ROWS and 0 <= mid_c < COLS): continue
            if not (0 <= new_r < ROWS and 0 <= new_c < COLS): continue
            if is_blue(mid_r, mid_c): continue
            if is_blue(new_r, new_c): continue
            moves.append((new_r, new_c, dname))
        graph[(r,c)] = moves

# Find connected components using simple BFS ignoring direction
from collections import deque
visited_component = set()
def get_component(start_cell):
    comp = set()
    q = deque([start_cell])
    while q:
        cell = q.popleft()
        if cell in comp: continue
        comp.add(cell)
        if cell in graph:
            for (nr, nc, _) in graph[cell]:
                if (nr,nc) not in comp:
                    q.append((nr,nc))
    return comp

comp_start = get_component(start)
comp_end = get_component(end)
print(f"\nComponent containing START ({chr(65+start[1])}{start[0]+1}): {len(comp_start)} cells")
print(f"Component containing END ({chr(65+end[1])}{end[0]+1}): {len(comp_end)} cells")
print(f"Are they the same component? {start in comp_end}")

print("\nCells in START component:")
for (r,c) in sorted(comp_start):
    print(f"  {chr(65+c)}{r+1} style={grid_styles[(r,c)]}")

print("\nCells in END component:")
for (r,c) in sorted(comp_end):
    print(f"  {chr(65+c)}{r+1} style={grid_styles[(r,c)]}")

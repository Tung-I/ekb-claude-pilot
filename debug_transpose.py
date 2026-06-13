
# Try transposed grid: rows become columns and vice versa
# If original is 20 rows x 9 cols, transposed is 9 rows x 20 cols
# START A1 -> row=0,col=0; transposed: row=0,col=0 same
# END I20 -> row=8,col=19 (I=col 8, 20=row 20 in 1-indexed -> col 19 in 0-indexed for transposed)

import re, zipfile
from collections import deque

xlsx_path = '/work/pi_rsitaram_umass_edu/tungi/ekb-claude-pilot/data/gaia/raw/65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx'
with zipfile.ZipFile(xlsx_path) as z:
    sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')

# Original grid_styles (row, col) in 0-indexed
orig_styles = {}
cell_pat = re.compile(r'<c r="([A-Z]+)(\d+)" s="(\d+)"')
for m in cell_pat.finditer(sheet):
    col_str, row_str, s = m.group(1), m.group(2), int(m.group(3))
    col = ord(col_str) - ord("A")
    row = int(row_str) - 1
    orig_styles[(row, col)] = s

ORIG_ROWS, ORIG_COLS = 20, 9

# Transposed: (row, col) -> (col, row)
trans_styles = {}
for (r, c), s in orig_styles.items():
    trans_styles[(c, r)] = s

TRANS_ROWS, TRANS_COLS = ORIG_COLS, ORIG_ROWS  # 9 rows x 20 cols

# In transposed grid, START is still at (0,0), END is at (8, 19)
# A1 in original = (row=0, col=0) -> transposed = (col=0, row=0) = (0,0)
# I20 in original = (row=19, col=8) -> transposed = (col=8, row=19) = (8, 19)

for r in range(TRANS_ROWS):
    for c in range(TRANS_COLS):
        if (r,c) not in trans_styles:
            trans_styles[(r,c)] = 0

BLUE_STYLE = 5

print("Transposed grid (9 rows x 20 cols):")
for r in range(TRANS_ROWS):
    row_str = ""
    for c in range(TRANS_COLS):
        s = trans_styles.get((r,c), 0)
        if (r,c) == (0,0): row_str += "S"
        elif (r,c) == (8,19): row_str += "E"
        elif s == BLUE_STYLE: row_str += "X"
        else: row_str += str(s)
    print(f"  Row {r+1}: {row_str}")

# Check connectivity in transposed grid
def is_blue_t(r, c):
    return trans_styles.get((r,c), 0) == BLUE_STYLE

DIRS = [(-1,0,"UP"), (1,0,"DOWN"), (0,-1,"LEFT"), (0,1,"RIGHT")]

start_t = (0, 0)
end_t = (8, 19)

from collections import deque
def get_component(start_cell, is_blue_fn, rows, cols):
    comp = set()
    graph = {}
    for r in range(rows):
        for c in range(cols):
            if is_blue_fn(r,c): continue
            moves = []
            for dr, dc, dname in DIRS:
                mid_r, mid_c = r+dr, c+dc
                new_r, new_c = r+2*dr, c+2*dc
                if not (0 <= mid_r < rows and 0 <= mid_c < cols): continue
                if not (0 <= new_r < rows and 0 <= new_c < cols): continue
                if not is_blue_fn(new_r, new_c):
                    moves.append((new_r, new_c))
            graph[(r,c)] = moves
    
    q = deque([start_cell])
    while q:
        cell = q.popleft()
        if cell in comp: continue
        comp.add(cell)
        for nb in graph.get(cell, []):
            if nb not in comp:
                q.append(nb)
    return comp

comp_start = get_component(start_t, is_blue_t, TRANS_ROWS, TRANS_COLS)
print(f"\nTransposed: Component containing START: {len(comp_start)} cells")
print(f"END in start component: {end_t in comp_start}")

# Print component cells
for c in sorted(comp_start):
    print(f"  {c} style={trans_styles[c]}")

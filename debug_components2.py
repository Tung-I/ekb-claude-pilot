
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
        if (r, c) not in grid_styles:
            grid_styles[(r, c)] = 0

BLUE_STYLE = 5
def is_blue(r, c):
    return grid_styles.get((r, c), 0) == BLUE_STYLE

start = (0, 0)
end = (19, 8)
DIRS = [(-1,0), (1,0), (0,-1), (0,1)]

# No intermediate check, no backward constraint: just 2-step moves between non-blue cells
graph = {}
non_blue = [(r,c) for r in range(ROWS) for c in range(COLS) if not is_blue(r,c)]
for r, c in non_blue:
    nbrs = []
    for dr, dc in DIRS:
        nr, nc = r + 2*dr, c + 2*dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and not is_blue(nr, nc):
            nbrs.append((nr, nc))
    graph[(r,c)] = nbrs

# Find all connected components
all_comps = []
visited = set()
for cell in non_blue:
    if cell in visited:
        continue
    comp = set()
    q = deque([cell])
    while q:
        cur = q.popleft()
        if cur in comp: continue
        comp.add(cur)
        for nb in graph.get(cur, []):
            if nb not in comp:
                q.append(nb)
    all_comps.append(comp)
    visited.update(comp)

print(f"Number of connected components: {len(all_comps)}")
for i, comp in enumerate(sorted(all_comps, key=len, reverse=True)):
    has_start = start in comp
    has_end = end in comp
    cells = sorted([f"{chr(65+c)}{r+1}" for r,c in comp])
    print(f"  Component {i+1} ({len(comp)} cells, start={has_start}, end={has_end}): {cells[:10]}{'...' if len(cells)>10 else ''}")

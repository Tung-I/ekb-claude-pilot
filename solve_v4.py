
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

# Style color mapping (the actual rgb hex without alpha)
style_rgb = {
    0: "FFFFFF",   # default/unset
    1: "FFFFFF",   # white (START/END)
    2: "92D050",   # green
    3: "FFFF00",   # yellow
    4: "F478A7",   # pink
    5: "0099FF",   # blue BLOCKED
}

BLUE_STYLE = 5
def is_blue(r, c):
    return grid_styles.get((r,c), 0) == BLUE_STYLE

start = (0, 0)
end = (19, 8)

DIRS = [(-1,0,"UP"), (1,0,"DOWN"), (0,-1,"LEFT"), (0,1,"RIGHT")]
OPPOSITE = {"UP":"DOWN", "DOWN":"UP", "LEFT":"RIGHT", "RIGHT":"LEFT"}

# Check: which cells can reach END (I20)?
print("Cells that can reach I20 (ignoring intermediate check):")
r_end, c_end = end
for dr, dc, dname in DIRS:
    # Move comes from opposite direction: start is at (r_end - 2*dr, c_end - 2*dc)
    src_r, src_c = r_end - 2*dr, c_end - 2*dc
    mid_r, mid_c = r_end - dr, c_end - dc
    if 0 <= src_r < ROWS and 0 <= src_c < COLS and 0 <= mid_r < ROWS and 0 <= mid_c < COLS:
        print(f"  From {chr(65+src_c)}{src_r+1} (style={grid_styles.get((src_r,src_c),0)}) via {chr(65+mid_c)}{mid_r+1} (style={grid_styles.get((mid_r,mid_c),0)}) -- src_blue={is_blue(src_r,src_c)}, mid_blue={is_blue(mid_r,mid_c)}")

# Now solve with NO intermediate check (can jump over blue)
def bfs_no_intermediate(no_backward, max_turns=60):
    queue = deque()
    queue.append((start[0], start[1], None, [start]))
    seen = set()
    solutions = []
    while queue:
        r, c, last_dir, path = queue.popleft()
        if len(path) > max_turns + 1:
            continue
        key = (r, c, last_dir, frozenset(path))
        if key in seen:
            continue
        seen.add(key)
        path_set = set(path)
        for dr, dc, dname in DIRS:
            if no_backward and last_dir and dname == OPPOSITE[last_dir]:
                continue
            new_r, new_c = r + 2*dr, c + 2*dc
            if not (0 <= new_r < ROWS and 0 <= new_c < COLS): continue
            if is_blue(new_r, new_c): continue
            if (new_r, new_c) in path_set: continue
            new_path = path + [(new_r, new_c)]
            if (new_r, new_c) == end:
                solutions.append(new_path)
                cells = [f"{chr(65+cc)}{rr+1}" for rr,cc in new_path]
                print(f"SOLUTION ({len(new_path)-1} turns, no_backward={no_backward}): {cells}")
                if len(solutions) >= 3:
                    return solutions
            else:
                queue.append((new_r, new_c, dname, new_path))
    return solutions

print("\n=== No intermediate check, no backward ===")
sols = bfs_no_intermediate(True, 60)
if not sols:
    print("\n=== No intermediate check, allow backward ===")
    sols = bfs_no_intermediate(False, 60)

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
        print(f"\nTURN 11 ANSWER: {chr(65+c11)}{r11+1}, color={clr}")
    else:
        print(f"Path only {len(path)-1} turns long!")
else:
    print("NO SOLUTION FOUND")

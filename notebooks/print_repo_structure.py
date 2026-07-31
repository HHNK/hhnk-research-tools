# %%
# Make a quick overview of the repository structure

import os
from pathlib import Path

repo = "business"
root = Path(rf"D:\github\wvangerwen\hhnk-hydro-{repo}\src\hhnk_hydro_{repo}")

# repo = "hhnk-threedi-tools"
# root = Path(rf"D:\github\wvangerwen\{repo}\{repo.replace('-', '_')}")

EXCLUDE_DIRS = {"__pycache__", ".mypy_cache", ".ipynb_checkpoints"}
for dirpath, dirnames, filenames in os.walk(root, topdown=True):
    dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

    level = Path(dirpath).relative_to(root).parts
    indent = " " * (len(level) - 1)

    # print the folder
    print(f"{indent} 📁{Path(dirpath).name}/")

    # print files in this folder
    for f in sorted(filenames):
        if "__init__.py" in f:
            continue

        # Get line count
        file_path = Path(dirpath).joinpath(f)
        with open(file_path, "r", encoding="utf-8") as file:
            line_count = sum(1 for _ in file)

        print(f"{indent}  📄{f}({line_count})")

# %%
import os
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

repo = "qgis"
root = Path(rf"D:\github\wvangerwen\hhnk-hydro-{repo}\src\hhnk_hydro_{repo}")

# -----------------------------
# Collect rows using your logic
# -----------------------------
# rows: (name, level, type)
rows = []

for dirpath, dirnames, filenames in os.walk(root, topdown=True):
    dirname = Path(dirpath).name
    if dirname == "__pycache__":
        continue

    # level: 0 for root, 1 for direct child, etc.
    level = len(Path(dirpath).relative_to(root).parts)

    # folder row
    rows.append((dirname + "/", level, "folder"))

    # file rows
    for f in sorted(filenames):
        if f == "__init__.py":
            continue
        rows.append((f, level + 1, "file"))

# -----------------------------
# Write to Excel
# -----------------------------
wb = Workbook()
ws = wb.active
ws.title = "Folder Tree"

ws.append(["Name", "Level", "Type"])
ws.freeze_panes = "A2"

for name, level, typ in rows:
    ws.append([name, level, typ])

# -----------------------------
# Formatting: indentation + bold
# -----------------------------
for row in ws.iter_rows(min_row=2):
    name_cell = row[0]
    level_cell = row[1]
    type_cell = row[2]

    level = level_cell.value

    # visual indent
    name_cell.alignment = Alignment(indent=level)

    # bold folders
    if type_cell.value == "folder":
        name_cell.font = Font(bold=True)

# -----------------------------
# Grouping: per-folder descendants
# -----------------------------
n_rows = len(rows)

for idx, (name, level, typ) in enumerate(rows):
    if typ != "folder":
        continue

    excel_row = idx + 2  # +2 because header is row 1 and rows list is 0-based

    # find the last descendant row for this folder
    start = excel_row + 1  # first child row (if any)
    end = start - 1  # will move forward if there are children

    for j in range(idx + 1, n_rows):
        child_name, child_level, _ = rows[j]

        # stop when we hit a row at same or higher-level folder
        if child_level <= level:
            break

        end = j + 2  # Excel row index

    # only group if we found children
    if end >= start:
        # outline level in Excel can mirror folder depth or depth+1;
        # here we use level + 1 so children visually nest nicely
        ws.row_dimensions.group(start, end, outline_level=level + 1)

ws.sheet_properties.outlinePr.summaryBelow = True

# -----------------------------
# Save workbook
# -----------------------------
wb.save("folder_tree.xlsx")

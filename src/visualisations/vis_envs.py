from src.custom_envs import *

Grids = [
    EasyDoorGrid,
    # EasyMuseumGrid,
    EmptyDirtyRoom,
    # EmptyGrid1D,
    MuseumRush,
    RandomMuseumRoom,
    # SimpleGrid,
    SmallMuseumGrid,
    SushiGrid
]

print("""
\\begin{table}[]
\\begin{tabular}{ll}
""")
for AGrid in Grids:
    grid = AGrid()
    grid.reset()
    open = str(grid.__class__.__name__) + " & " + "\\begin{tabular}[c]{@{}l@{}}"
    mid = np_grid_to_string(grid._get_grid_with_rob(),
                            should_color=False,
                            should_emojify=True).replace("\n", "\\\\ ", 100)
    close = "\end{tabular} \\\\"
    print(open + mid + close)
print("""
\\end{tabular}
\\end{table})
""")

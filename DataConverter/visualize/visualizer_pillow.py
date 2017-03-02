def smallest_size(images):
    smallest = -1
    smallest_idx = -1
    smallest_col = -1
    smallest_row = -1
    smallest_col_idx = -1
    smallest_row_idx = -1

    for idx, image in enumerate(images):
        imgcol = image.size[0] # todo col or row ??
        imgrow = image.size[1]
        imgsize = imgcol * imgrow

        if smallest == -1:
            smallest = imgsize
            smallest_col = imgcol
            smallest_row = imgrow
            smallest_idx = idx
            smallest_col_idx = idx
            smallest_row_idx = idx
            continue

        if smallest > imgsize:
            smallest = imgsize
            smallest_idx = idx

        if smallest_col > imgcol:
            smallest_col = imgcol
            smallest_col_idx = idx

        if smallest_row > imgrow:
            smallest_row = imgrow
            smallest_row_idx = idx

    return [smallest, smallest_idx], [smallest_col, smallest_col_idx], [smallest_row, smallest_row_idx]

def average_size(images):
    sum_col = 0
    sum_row = 0
    for image in images:
        sum_col += image.size[0] # todo col or row??
        sum_row += image.size[1]

    avg_col = sum_col / len(images)
    avg_row = sum_row / len(images)
    return avg_col, avg_row

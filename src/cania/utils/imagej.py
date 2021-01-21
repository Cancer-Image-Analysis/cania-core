
def read_elipses_from_csv(dataframe):
    ellipses = []
    scale = 1.
    for _, ellipse in dataframe.iterrows():
        ellipse_info = read_elipses_from_row(ellipse, scale=scale)
        ellipses.append(ellipse_info)
    return ellipses

def read_elipses_from_row(row, scale=1.):
    angle = 180 - row.Angle
    position = (row.X*scale, row.Y*scale)
    size = (row.Major*scale, row.Minor*scale)
    ellipse_info = (position, size, angle)
    print(ellipse_info)
    return ellipse_info
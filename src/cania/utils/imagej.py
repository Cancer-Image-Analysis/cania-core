
def read_elipses_from_csv(dataframe):
    ellipses = []
    s = 1.
    for _, ellipse in dataframe.iterrows():
        angle = 180 - ellipse.Angle
        position = (ellipse.X*s, ellipse.Y*s)
        size = (ellipse.Major*s, ellipse.Minor*s)
        ellipse_info = (position, size, angle)
        ellipses.append(ellipse_info)
    return ellipses

from PIL import Image, ImageDraw, ImageFont
from itertools import product
import numpy as np
from .frame import REAL_NROWS, REAL_NCOLS

class Painter(object):
    def __init__(self, nrows=REAL_NROWS, ncols=REAL_NCOLS) -> None:
        """
        Painter class is used to generate coordinates of patterns on a rectangular grid.
        The "draw" functions return the coordinates of the points in the pattern.
        --------------------
        Parameters:
        --------------------
        nrows: int
            Number of rows in the rectangular grid
        ncols: int
            Number of columns in the rectangular grid
        """
        self.nrows = nrows
        self.ncols = ncols
    
    def parseRange(self, x):
        """
        Parse the range argument to a valid range
        --------------------
        Parameters:
        --------------------
        x: int | array-like | range | list
            int for number of elements, array-like for list of elements

        --------------------
        Returns:
        --------------------
        x: array-like
            array-like of shape (N,)
        """
        if isinstance(x, int):
            assert x >= 0, 'x must be a non-negative integer'
            x = list(range(x))
        elif isinstance(x, range):
            x = list(x)
        elif x is None:
            x = list(range(-100, 100))
        return x

    def drawText(self, 
                text='A', 
                offset=(0, 0),
                font_size=500, 
                stroke_width=0,
                font='arial.ttf',
                rotate=45):
        """
        Draw text on the rectangular grid
        --------------------
        Parameters:
        --------------------
        letter: str
            Letter to be drawn
        offset: tuple
            Offset of the center of the letter with respect to the center of the grid
        font_size: int
            Font size of the letter
        stroke_width: int
            Width of the stroke of the letter
        font: str
            Font of the letter
        rotate: float
            Angle to rotate the letter

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the letter
        """
        row, col = self.nrows // 2 + offset[0], self.ncols // 2 + offset[1]
        
        # Build a transparent image large enough to hold the text
        mask = Image.new('L', (self.nrows, self.ncols), 0)
        draw = ImageDraw.Draw(mask)
        font = ImageFont.truetype(font, font_size)
        draw.text((row, col), text=text, font=font, fill=255, 
                  stroke_width=stroke_width, anchor='mm')
        mask = mask.rotate(rotate)

        # Find the coordinates of the points in the letter
        return np.argwhere(np.array(mask) == 255)
    
    def drawCircle(self, 
                   row_offset=0, 
                   col_offset=0, 
                   radius=50):
        """
        Draw a filled circle on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the circle
        col_offset: int
            Column offset of the center of the circle
        radius: int
            Radius of the circle

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the circle
        """
        # Find the center coordinates
        center_row, center_col = self.nrows // 2, self.ncols // 2
        row, col = center_row + row_offset, center_col + col_offset

        # Draw a filled circle with the given radius
        ans = [(i, j) for i in range(max(0, int(row - radius)), min(int(row + radius + 1), self.nrows))\
                    for j in range(max(0, int(col - radius)), min(int(col + radius + 1), self.ncols)) \
                    if (i-row)**2 + (j-col)**2 <= radius**2]
        return np.array(ans)
    
    def drawCircleOutline(self,
                            row_offset=0,
                            col_offset=0,
                            radius1=50,
                            radius2=40):
        """
        Draw an outlined circle on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the circle
        col_offset: int
            Column offset of the center of the circle
        radius1: int
            Outer radius of the circle
        radius2: int
            Inner radius of the circle

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the outlined circle
        """
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset
        ans = [(i, j) for i in range(max(0, int(center_row - radius1)), min(int(center_row + radius1 + 1), self.nrows))\
                for j in range(max(0, int(center_col - radius1)), min(int(center_col + radius1 + 1), self.ncols))\
                if (i-center_row)**2 + (j-center_col)**2 <= radius1**2 and (i-center_row)**2 + (j-center_col)**2 >= radius2**2]
        return np.array(ans)
    
    def drawSquare(self, 
                   radius=3, 
                   row_offset=0, 
                   col_offset=0):
        """
        Draw a square on the rectangular grid
        --------------------
        Parameters:
        --------------------
        radius: int
            Radius of the square
        row_offset: int
            Row offset of the center of the square
        col_offset: int
            Column offset of the center of the square

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the square
        """
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset
        ans = [(i, j) for i in range(max(0, int(center_row - radius)), min(int(center_row + radius + 1), self.nrows))\
                for j in range(max(0, int(center_col - radius)), min(int(center_col + radius + 1), self.ncols))]
        return np.array(ans)
    
    def drawAnchorCircles(self,
                          anchor=((0, 0), (200, 0), (0, 250)),
                          radius=10):
        """
        Draw anchor circles on the rectangular grid
        --------------------
        Parameters:
        --------------------
        anchor: array-like of shape (N, 2)
            coordinates of the anchor circles
        radius: int
            radius of the anchor circles

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the anchor circles
        """
        corr = []
        for x, y in anchor:
            new_circle = self.drawCircle(row_offset=x, 
                                         col_offset=y, 
                                         radius=radius)
            if new_circle.shape[0] != 0: corr.append(new_circle)
        return np.concatenate(corr, axis=0)
    
    def drawAnchorCircleOutlines(self,
                                anchor=((0, 0), (200, 0), (0, 250)),
                                radius1=10,
                                radius2=5):
        """
        Draw anchor circle outlines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        anchor: array-like of shape (N, 2)
            coordinates of the anchor circles
        radius1: float
            outer radius of the anchor circles
        radius2: float
            inner radius of the anchor circles
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the anchor circle outlines
        """
        corr = []
        for x, y in anchor:
            new_circle = self.drawCircleOutline(row_offset=x, 
                                                col_offset=y, 
                                                radius1=radius1,
                                                radius2=radius2)
            if new_circle.shape[0] != 0: corr.append(new_circle)
        return np.concatenate(corr, axis=0)
    
    def drawAnchorSquares(self,
                            anchor=((0, 0), (200, 0), (0, 250)),
                            radius=10):
        """
        Draw anchor squares on the rectangular grid
        --------------------
        Parameters:
        --------------------
        anchor: array-like of shape (N, 2)
            coordinates of the anchor squares
        radius: int
            radius of the anchor squares

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the anchor squares
        """
        corr = []
        for x, y in anchor:
            new_square = self.drawSquare(row_offset=x, 
                                         col_offset=y, 
                                         radius=radius)
            if new_square.shape[0] != 0: corr.append(new_square)
        return np.concatenate(corr, axis=0)
    
    def drawArrayOfCircles(self, 
                          row_spacing=50, 
                          col_spacing=50, 
                          row_offset=0, 
                          col_offset=0, 
                          nx=None, 
                          ny=None, 
                          radius=1):
        """
        Draw an array of circles on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of circles
        col_spacing: int
            Spacing between columns of circles
        row_offset: int
            Row offset of the center of the first circle
        col_offset: int
            Column offset of the center of the first circle
        nx: int
            Number of circles in each row
        ny: int
            Number of circles in each column
        radius: int
            Radius of the circles

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the arrays of circles
        """
        return self.drawArrayOfCirclesAngled(angle=0,
                                            row_spacing=row_spacing,
                                            col_spacing=col_spacing,
                                            row_offset=row_offset,
                                            col_offset=col_offset,
                                            nx=nx,
                                            ny=ny,
                                            radius=radius)
    

    def drawArrayOfCirclesAngled(self,
                                angle=45,
                                row_spacing=50,
                                col_spacing=50,
                                row_offset=0,
                                col_offset=0,
                                nx=None,
                                ny=None,
                                radius=5):
        """
        Draw an array of circles on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the circles in degrees
        row_spacing: int
            Spacing between rows of circles
        col_spacing: int    
            Spacing between columns of circles
        row_offset: int
            Row offset of the center of the first circle
        col_offset: int
            Column offset of the center of the first circle
        nx: int
            Number of circles in each row
        ny: int
            Number of circles in each column
        radius: int
            Radius of the circles

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the arrays of circles
        """
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        anchors = np.array([(i*row_spacing+row_offset, j*col_spacing+col_offset) 
                            for i, j in product(nx, ny)])
        rotation_matrix = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
        anchors = anchors @ rotation_matrix
        return self.drawAnchorCircles(anchor=anchors, radius=radius)
    
    def drawArrayOfCirclesLattice(self, 
                                  spacing=100,
                                  row_offset=0,
                                  col_offset=0,
                                  radius=5,
                                  angle1=0,
                                  angle2=90,
                                  nx=None,
                                  ny=None):
        """
        Draw an array of circles on the rectangular grid with lattice structure
        --------------------
        Parameters:
        --------------------
        spacing: int
            Spacing between the circles
        row_offset: int
            Row offset of the center of the first circle
        col_offset: int
            Column offset of the center of the first circle
        radius: int
            Radius of the circles
        angle1: float
            Angle of the first lattice vector in degrees
        angle2: float   
            Angle of the second lattice vector in degrees

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the arrays of circles
        """
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        corr = []
        a1, a2 = np.deg2rad(angle1), np.deg2rad(angle2)
        for i, j in product(nx, ny):
            new_circle = self.drawCircle(row_offset=i*spacing*np.cos(a1)+j*spacing*np.cos(a2)+row_offset, 
                                         col_offset=i*spacing*np.sin(a1)+j*spacing*np.sin(a2)+col_offset,
                                         radius=radius)
            if new_circle.shape[0] != 0: corr.append(new_circle)
        return np.concatenate(corr, axis=0)
    
    def drawArrayOfSquares(self, 
                        row_spacing=50, 
                        col_spacing=50, 
                        row_offset=0, 
                        col_offset=0, 
                        nx=None, 
                        ny=None, 
                        radius=3):
        """
        Draw an array of squares on the rectangular grid

        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of squares
        col_spacing: int
            Spacing between columns of squares
        row_offset: int
            Row offset of the center of the first square
        col_offset: int
            Column offset of the center of the first square
        nx: int
            Number of squares in each row
        ny: int
            Number of squares in each column
        radius: int
            Radius of the squares

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of squares
        """
        return self.drawArrayOfSquaresAngled(angle=0,
                                            row_spacing=row_spacing,
                                            col_spacing=col_spacing,
                                            row_offset=row_offset,
                                            col_offset=col_offset,
                                            nx=nx,
                                            ny=ny,
                                            radius=radius)

    def drawArrayOfSquaresAngled(self,
                                angle=45,
                                row_spacing=50,
                                col_spacing=50,
                                row_offset=0,
                                col_offset=0,
                                nx=None,
                                ny=None,
                                radius=3):
        """
        Draw an array of squares on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the squares in degrees
        row_spacing: int
            Spacing between rows of squares
        col_spacing: int
            Spacing between columns of squares
        row_offset: int
            Row offset of the center of the first square
        col_offset: int
            Column offset of the center of the first square
        nx: int
            Number of squares in each row
        ny: int
            Number of squares in each column
        radius: int
            Radius of the squares

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of squares
        """
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        anchors = np.array([(i*row_spacing+row_offset, j*col_spacing+col_offset) 
                            for i, j in product(nx, ny)])
        rotation_matrix = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
        anchors = anchors @ rotation_matrix
        return self.drawAnchorSquares(anchor=anchors, radius=radius)

    def drawHorizontalLine(self, 
                           row_offset=0, 
                           width=1,
                           center=False):
        """
        Draw a horizontal line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the line
        width: int
            Width of the line
        center: bool
            True to draw the line at the offseted center of the grid, False otherwise
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        row = self.nrows // 2 + int(row_offset)
        if center: 
            line_range = range(max(0, row - width // 2), min(self.nrows, row + width // 2 + 1))
        else:
            line_range = range(max(0, row), min(self.nrows, row + width))
        ans = np.array([(i, j) for j in range(self.ncols) for i in line_range])
        return ans
    
    def drawVerticalLine(self, 
                         col_offset=0, 
                         width=1, 
                         center=False):
        """
        Draw a vertical line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_offset: int
            Column offset of the center of the line
        width: int
            Width of the line
        center: bool
            True to draw the line at the offseted center of the grid, False otherwise
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        col = self.ncols // 2 + int(col_offset)
        if center: 
            line_range = range(max(0, col - width // 2), min(self.ncols, col + width // 2 + 1))
        else:
            line_range = range(max(0, col), min(self.ncols, col + width))
        ans = np.array([(i, j) for i in range(self.nrows) for j in line_range])
        return ans
    
    def drawLineABC(self, A, B, C, d):
        """
        Draw a line with equation Ax + By + C = 0 on the rectangular grid
        --------------------
        Parameters:
        --------------------
        A: float
            Coefficient of x in the line equation
        B: float
            Coefficient of y in the line equation
        C: float
            Constant term in the line equation
        d: float
            Width of the line

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        mask = (np.abs(A * cols + B * rows + C) <= d).astype(bool).flatten()
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()

    def drawAngledLine(self, 
                       angle=45, 
                       offset=0,
                       row_offset=None,
                       col_offset=None,
                       width=10,
                       center=False):
        """
        Draw an angled line on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the line in degrees
        offset: float
            Offset of the center of the line, optional if row_offset and col_offset are given
        row_offset: float
            Row offset of the center of the line, optional if offset is given
        col_offset: float
            Column offset of the center of the line, optional if offset is given
        width: float
            Width of the line

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the line
        """
        # Find the center coordinates
        angle = np.deg2rad(angle % 180)
        if row_offset is not None and col_offset is not None:
            center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset
        else:
            center_row, center_col = self.nrows // 2 - offset * np.cos(angle), self.ncols // 2 + offset * np.sin(angle)

        # Draw a line with the given angle
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        dist = (cols - center_col) * np.sin(angle) - (rows - center_row) * np.cos(angle)
        if center:
            mask = (np.abs(dist) <= width // 2).astype(bool).flatten()
        else:
            mask = ((dist >= 0) & (dist <= width)).astype(bool).flatten()
        
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()
    
    def drawAngledLinesBundle(self,
                        angle=[0, 45, 90, 135],
                        row_offset=0,
                        col_offset=0,
                        width=10):
        """
        Draw an array of angled lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: array-like
            Angle of the lines in degrees
        row_offset: int
            Row offset of the center of the lines
        col_offset: int
            Column offset of the center of the lines
        width: int
            Width of the lines

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        angle = self.parseRange(angle)
        corr = []
        for a in angle:
            new_line = self.drawAngledLine(angle=a, 
                                           offset=row_offset * np.sin(np.deg2rad(a)) - col_offset * np.cos(np.deg2rad(a)), 
                                           width=width, center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)

    def drawAngledLines(self,
                        angle=45,
                        spacing=100,
                        offset=0,
                        row_offset=None,
                        col_offset=None,
                        width=10,
                        nx=5):
        """
        Draw an array of parallel lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the lines in degrees
        spacing: int
            Spacing between the lines
        offset: float
            Offset of the center of the first line, optional if row_offset and col_offset are given
        row_offset: float
            Row offset of the center of the lines, optional if offset is given
        col_offset: float
            Column offset of the center of the lines, optional if offset is given
        width: int
            Width of the lines
        nx: int | array-like
            Number of lines, or a list of row indices to draw lines on

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        nx = self.parseRange(nx)
        corr = []
        if row_offset is not None and col_offset is not None:
            offset = row_offset * np.sin(np.deg2rad(angle)) - col_offset * np.cos(np.deg2rad(angle))
        for i in nx:
            new_line = self.drawAngledLine(angle=angle, 
                                           offset=i*spacing+offset,
                                           width=width, center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)
    
    def drawAngledLinesOffset(self,
                              angle=45,
                              offset=[0],
                              width=10):
        """
        Draw an array of parallel lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the lines in degrees
        offset: array-like
            Offset of the center of the lines
        width: int
            Width of the lines

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        corr = []
        for ofs in offset:
            new_line = self.drawAngledLine(angle=angle, 
                                           offset=ofs,
                                           width=width, center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)

    def drawCross(self, 
                  row_offset=0, 
                  col_offset=0, 
                  width=1):
        """
        Draw a cross on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the cross
        col_offset: int
            Column offset of the center of the cross
        width: int
            Width of the lines in the cross

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the cross
        """
        return np.concatenate((self.drawHorizontalLine(row_offset=row_offset, width=width, center=True),
                               self.drawVerticalLine(col_offset=col_offset, width=width, center=True)), axis=0)
    
    def drawHorizontalLines(self, 
                            row_spacing=100, 
                            row_offset=0, 
                            width=1, 
                            ny=5):
        """
        Draw an array of horizontal lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of lines
        row_offset: int
            Row offset of the center of the first line
        width: int
            Width of the lines
        ny: int | array-like
            Number of lines, or a list of row indices to draw lines on
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        ny = self.parseRange(ny)
        corr = []
        for i in ny:
            new_line = self.drawHorizontalLine(row_offset=i*row_spacing+row_offset, 
                                        width=width,
                                        center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)
    
    def drawVerticalLines(self, 
                          col_spacing=100, 
                          col_offset=0, 
                          width=1, 
                          nx=5):
        """
        Draw an array of vertical lines on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_spacing: int
            Spacing between columns of lines
        col_offset: int
            Column offset of the center of the first line
        width: int
            Width of the lines
        nx: int | array-like
            Number of lines, or a list of column indices to draw lines on
                  
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of lines
        """
        nx = self.parseRange(nx)
        corr = []
        for j in nx:
            new_line = self.drawVerticalLine(col_offset=j*col_spacing+col_offset, 
                                        width=width,
                                        center=True)
            if new_line.shape[0] != 0: corr.append(new_line)
        return np.concatenate(corr, axis=0)
    
    def drawCrosses(self, 
                    row_spacing=100, 
                    col_spacing=100, 
                    row_offset=0, 
                    col_offset=0, 
                    width=1, 
                    nx=5, 
                    ny=5):
        """
        Draw an array of crosses on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_spacing: int
            Spacing between rows of crosses
        col_spacing: int
            Spacing between columns of crosses
        row_offset: int
            Row offset of the center of the first cross
        col_offset: int
            Column offset of the center of the first cross
        width: int
            Width of the lines in the crosses
        nx: int
            Number of crosses in each row
        ny: int
            Number of crosses in each column
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the array of crosses
        """
        nx = self.parseRange(nx)
        ny = self.parseRange(ny)
        corr = [self.drawHorizontalLines(row_spacing=row_spacing, row_offset=row_offset, width=width, ny=ny),
                self.drawVerticalLines(col_spacing=col_spacing, col_offset=col_offset, width=width, nx=nx)]
        return np.concatenate(corr, axis=0)
    
    def drawAngledCross(self,
                        angle=45,
                        angle2=None,
                        row_offset=0,
                        col_offset=0,
                        width=10):
        """
        Draw an angled cross on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the cross in degrees
        angle2: float
            Angle of the second line of the cross in degrees
        row_offset: int
            Row offset of the center of the cross
        col_offset: int
            Column offset of the center of the cross
        width: int
            Width of the lines in the cross

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the cross
        """
        if angle2 is None: angle2 = angle + 90
        return np.concatenate((self.drawAngledLine(angle=angle, row_offset=row_offset, col_offset=col_offset, width=width, center=True),
                               self.drawAngledLine(angle=angle2, row_offset=row_offset, col_offset=col_offset, width=width, center=True)), axis=0)

    def drawStar(self, 
                 row_offset=0, 
                 col_offset=0, 
                 num=10):
        """
        Draw a star on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            Row offset of the center of the star
        col_offset: int
            Column offset of the center of the star
        num: int
            Number of different sectors in the star

        --------------------
        Returns:
        --------------------
        corr: int
            Coordinates of the points in the star
        """
        # Find the center coordinates
        center_row, center_col = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset

        # Draw a star with the given number of sectors
        angle = 2 * np.pi / num
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        mask = ((np.arctan2(cols.flatten() - center_col, rows.flatten() - center_row) // angle) % 2).astype(bool)
        
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()
    
    def drawCheckerBoard(self, size=20):
        """
        Draw a checker board on the rectangular grid

        --------------------
        Parameters:
        --------------------
        size: int
            Side length of one checker board square
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the checker board
        """
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        mask = (((rows.flatten() // size) % 2 + (cols.flatten() // size) % 2) % 2).astype(bool)
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()
    
    def drawHorizontalHalfPlane(self,
                                row_offset=0):
        """
        Draw a horizontal half plane on the rectangular grid
        --------------------
        Parameters:
        --------------------
        row_offset: int
            row offset of the half plane
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the half plane
        """
        center_row = self.nrows // 2 + row_offset
        assert center_row >= 0 and center_row < self.nrows, 'Row offset out of range'
        ans = np.array([(i, j) for i in range(center_row, self.nrows) for j in range(self.ncols)])
        return ans

    def drawVerticalHalfPlane(self,
                              col_offset=0):
        """
        Draw a vertical half plane on the rectangular grid
        --------------------
        Parameters:
        --------------------
        col_offset: int
            column offset of the half plane

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the half plane
        """
        center_col = self.ncols // 2 + col_offset
        assert center_col >= 0 and center_col < self.ncols, 'Column offset out of range'
        ans = np.array([(i, j) for i in range(self.nrows) for j in range(center_col, self.ncols)])
        return ans
    
    def drawAngledHalfPlane(self,
                            angle=45,
                            row_offset=0,
                            col_offset=0):
        """
        Draw an angled half plane on the rectangular grid
        --------------------
        Parameters:
        --------------------
        angle: float
            Angle of the half plane in degrees
        row_offset: int
            row offset of the half plane
        col_offset: int
            column offset of the half plane

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 2)
            Coordinates of the points in the half plane
        """
        angle = angle % 180
        if angle == 0:
            return self.drawHorizontalHalfPlane(row_offset=row_offset)
        elif angle == 90:
            return self.drawVerticalHalfPlane(col_offset=col_offset)
        i, j = self.nrows // 2 + row_offset, self.ncols // 2 + col_offset
        angle = np.deg2rad(angle)
        rows, cols = np.meshgrid(np.arange(self.nrows), np.arange(self.ncols), indexing='ij')
        dist = (cols - j) * np.sin(angle) - (rows - i) * np.cos(angle)
        mask = (dist >= 0).astype(bool).flatten()
        return np.stack((rows.flatten()[mask], cols.flatten()[mask])).transpose()

    

class GrayscalePainter(Painter):
    def __init__(self, nrows=REAL_NROWS, ncols=REAL_NCOLS) -> None:
        super().__init__(nrows, ncols)
        self.rows, self.cols = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
        self.rows, self.cols = self.rows.flatten().astype(int), self.cols.flatten().astype(int)
    
    def normalizePattern(self, image: np.ndarray):
        """
        Normalize the stored pattern to [0, 1], works in-place
        """
        max_val = image.max()
        min_val = image.min()
        if max_val == min_val:
            image.fill(0)
        else:
            np.copyto(image, (image - min_val) / (max_val - min_val))

    def checkVector(self, vector):
        """
        Check if the given vector is a valid lattice vector
        --------------------
        Parameters:
        --------------------
        vector: list | array-like
            Lattice vector to be checked

        --------------------
        Raises:
        --------------------
        ValueError: if the given vector is not a valid lattice vector
        """
        if isinstance(vector, list):
            assert len(vector) == 2, 'Lattice vector 1 must be a list of length 2'
        elif isinstance(vector, np.ndarray):
            assert vector.shape == (2,), 'Lattice vector 1 must be an array of shape (2,)'
        else:
            raise ValueError('Lattice vector 1 must be a list or numpy array of shape (2,)')

    def draw1dLattice(self,
                      lat_vec=[0.01, 0.01],
                      x_offset=0,
                      y_offset=0,
                      ):
        """
        Draw a 1D lattice on the rectangular grid. The intensity is given by cos(2*\pi*k*x) where k is the lattice vector.
        --------------------
        Parameters:
        --------------------
        lat_vec: array-like of shape (2,)
            Lattice vector of the lattice
        
        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 3)
            Coordinates of the points in the lattice, the third column as the intensity
        """
        self.checkVector(lat_vec)
        center_row, center_col = self.nrows // 2 + x_offset, self.ncols // 2 + y_offset
        pattern = np.cos(2 * np.pi * (lat_vec[0]*(self.rows - center_row) + lat_vec[1]*(self.cols - center_col)))
        self.normalizePattern(pattern)
        return np.stack((self.rows, self.cols, pattern.flatten())).transpose()
    
    def draw2dLattice(self,
                      lat_vec1 = [0.01, 0.],
                      lat_vec2 = [0., 0.01],
                      x_offset=0, 
                      y_offset=0,
                      interference=False,
                      ):
        """
        Draw a 2D lattice on the rectangular grid. The intensity is given by cos(k1*x) + cos(k2*x) + {cos((k1-k2)*x)} where k1, k2 is the lattice vector.
        --------------------
        Parameters:
        --------------------
        lat_vec1: array-like of shape (2,)
            Lattice vector of the first lattice beam
        lat_vec2: array-like of shape (2,)
            Lattice vector of the second lattice beam
        x_offset: int
            Row offset of the center of the lattice
        y_offset: int
            Column offset of the center of the lattice
        interference: bool
            True to add interference term to the lattice

        --------------------
        Returns:
        --------------------
        corr: array-like of shape (N, 3)
            Coordinates of the points in the lattice, the third column as the intensity
        """
        self.checkVector(lat_vec1)
        self.checkVector(lat_vec2)
        center_row, center_col = self.nrows // 2 + x_offset, self.ncols // 2 + y_offset

        if not interference:
            pattern = np.cos(2 * np.pi * (lat_vec1[0]*(self.rows - center_row) + lat_vec1[1]*(self.cols - center_col))) + \
                        np.cos(2 * np.pi * (lat_vec2[0]*(self.rows - center_row) + lat_vec2[1]*(self.cols - center_col)))
        else:
            pattern = np.cos(2 * np.pi * (lat_vec1[0]*(self.rows - center_row) + lat_vec1[1]*(self.cols - center_col))) + \
                        np.cos(2 * np.pi * (lat_vec2[0]*(self.rows - center_row) + lat_vec2[1]*(self.cols - center_col))) + \
                        np.cos(2 * np.pi * ((lat_vec1[0] - lat_vec2[0])*(self.rows - center_row) + (lat_vec1[1] - lat_vec2[1])*(self.cols - center_col)))
        self.normalizePattern(pattern)
        return np.stack((self.rows, self.cols, pattern.flatten())).transpose()

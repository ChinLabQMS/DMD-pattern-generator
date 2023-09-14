from PIL import Image

def circle_bright(template, col, row, radius=50):
    assert isinstance(template, Image.Image)

    for i in range(max(0, row-radius), min(row+radius+1, template.size[1])):
        for j in range(max(0, col-radius), min(col+radius+1, template.size[0])):

            if (i-row)**2 + (j-col)**2 <= radius**2:
                template.putpixel((i, j), value=1)
    
    return template
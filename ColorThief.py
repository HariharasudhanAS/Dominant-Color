from colorthief import ColorThief

color_thief = ColorThief('/home/orange/Desktop/test/img0 20171230_123040.jpg')
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
# build a color palette
palette = color_thief.get_palette(color_count=4)
print (dominant_color)
print (palette)

# img = imread ("color.jpg")


import os
import cv2
import imageio as io

def _pad(im, total_width, color):
    imH, imW, _  = im.shape
    padding = (total_width - imW) // 2
    if padding < 0: return im

    im = cv2.copyMakeBorder(im, 0, 0, padding, padding, cv2.BORDER_CONSTANT, None, value = color)
    return im

def pad_image(im_file, total_width, color = (255, 255, 255)):
    file_path = os.path.dirname(im_file)
    base_name, ext = os.path.splitext(os.path.basename(im_file))

    if ext == ".gif":
        vr = io.get_reader(im_file)
        fps = 1000 / (vr.get_meta_data()['duration'] * 10 / len(vr))
        
        vw = io.get_writer(f"{file_path}/{base_name}_paddded{ext}", fps=fps)
        for im in vr:
            im = _pad(im, total_width, color)
            vw.append_data(im)

        vr.close()
        vw.close()
    else:
        im = io.imread(im_file)
        im = _pad(im, total_width, color)
        io.imwrite(f"{file_path}/{base_name}_paddded{ext}", im)


if __name__ == "__main__":
    pad_image("docs/dark-1/README/optimization-t.png", 1000, (255, 255, 255, 255))
    
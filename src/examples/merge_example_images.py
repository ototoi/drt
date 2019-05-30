import cv2
import os 
import glob
import argparse

import numpy as np


def process(args):
    idir = args.idir
    odir = args.odir
    s_img = cv2.imread(os.path.join(idir, 'start.png'))
    g_img = cv2.imread(os.path.join(idir, 'goal.png'))


    font = cv2.FONT_HERSHEY_PLAIN
	#文字の書き込み
    font_size = 5
    cv2.putText(s_img, 'Start', (200, 500), font, font_size, (255, 255, 255))
    cv2.putText(g_img, 'Goal', (200, 500), font, font_size, (255, 255, 255))

    h, w = s_img.shape[:2]
    t_img = np.zeros((h, 3*w, 3), dtype=np.uint8)
    t_img[:, w:2*w, :] = s_img
    t_img[:, 2*w: , :] = g_img
    ipaths = sorted(glob.glob(os.path.join(idir, '*.png')))
    os.makedirs(odir, exist_ok=True)
    for i in range(len(ipaths)):
        ipath = ipaths[i]
        img = cv2.imread(ipath)
        t_img[:, 0:w, :] = img
        opath = os.path.join(odir, '{0:08d}.png'.format(i))
        cv2.imwrite(opath, t_img)
    return 0



def main() -> int:
    parser = argparse.ArgumentParser(description='DRT')
    parser.add_argument(
        '--idir', '-g', default='./data/backprop_material_cornelbox/', help='output file directory path')
    parser.add_argument(
        '--odir', '-o', default='./output/backprop_material_cornelbox/', help='output file directory path')
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    exit(main())
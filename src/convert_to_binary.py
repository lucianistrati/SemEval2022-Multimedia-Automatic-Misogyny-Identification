from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable

import os
import cv2
import pdb


def main():
    path = "data/TRAINING"
    saving_path = "data/TRAINING_bytes"
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:

            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(dirpath, filename))
                is_success, im_buf_arr = cv2.imencode(".jpg", img)
                byte_img = im_buf_arr.tobytes()

                # print(type(byte_img))

                f = open(os.path.join(saving_path, filename[:filename.find(".")] + ".txt"), "wb")
                f.write(byte_img)
                f.close()


if __name__ == '__main__':
    main()

import numpy as np
import cv2


class VisualAPI(object):
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1
    @staticmethod
    def plot_vertices(image, vertices):
        image = image.copy()
        vertices = np.round(vertices).astype(np.int32)
        for i in range(0, vertices.shape[0], 2):
            st = vertices[i, :2]
            image = cv2.circle(image, (st[0], st[1]), 1, (255, 0, 0), -1)
        return image
    @classmethod
    def plot_kpt(cls, image, kpt):

        ''' Draw 68 key points
        Args:
            image: the input image
            kpt: (68, 3).
        '''
        image = image.copy()
        kpt = np.round(kpt).astype(np.int32)
        for i in range(kpt.shape[0]):
            st = kpt[i, :2]
            image = cv2.circle(image, (st[0], st[1]), 1, (0, 0, 255), 2)
            if i in cls.end_list:
                continue
            ed = kpt[i + 1, :2]
            image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
        return image
    @classmethod
    def write_kpt(cls, img, pos, face_ind):
        """
        #TODO remove
        only for PRNet
        :param cls:
        :param img:
        :param pos:
        :param face_ind:
        :return:
        """
        cls.count += 1
        all_vertices = np.reshape(pos*255, [256 ** 2, -1])
        vertices = all_vertices[face_ind, :]
        save_vertices = vertices.copy()
        save_vertices[:, 1] = 256 - 1 - save_vertices[:, 1]
        cv2.imwrite('../transform_sample/{}.jpg'.format(cls.count), cls.plot_vertices(img, vertices))







# pos = np.load(map_path)
# img = cv2.imread(img_path)
#
# all_vertices = np.reshape(pos, [256 ** 2, -1])
# vertices = all_vertices[face_ind, :]
#
# save_vertices = vertices.copy()
# save_vertices[:, 1] = 256 - 1 - save_vertices[:, 1]

# cv2.imwrite('{}.jpg'.format(0), plot_vertices(img, vertices))


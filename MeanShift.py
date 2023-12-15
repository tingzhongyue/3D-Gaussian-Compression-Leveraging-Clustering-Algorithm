from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
def get_Gaussiandata(file_dir):
    plydata = PlyData.read(file_dir + "point_cloud.ply")  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]
    print(data_np.shape)
    return data_np

def MeanShift(points):
    bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=100000)
    fc = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    fc.fit(points[:,6:54])
    return fc.cluster_centers_, fc.labels_

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def save_ply(file,Gaussian_datas):
    l = construct_list_of_attributes()
    print(len(l))
    xyz = Gaussian_datas[:,:3]
    normals = np.zeros_like(xyz)
    f_dc = Gaussian_datas[:,6:9]
    f_rest = Gaussian_datas[:,9:54]
    opacities = Gaussian_datas[:,54].reshape((-1,1))
    scale = Gaussian_datas[:,55:58]
    rotation = Gaussian_datas[:,58:62]
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(file)

def VQ_Gaussian(file,Gaussian_datas,centers,labels):
    file = file + 'point_cloud_vq.ply'
    for i in range(len(Gaussian_datas)):
        label = labels[i]
        Gaussian_datas[i,6:54] = centers[label]
    print(Gaussian_datas)
    save_ply(file,Gaussian_datas)

def construct_list_of_attributes_Compact():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz','label']
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def save_ply_Compact(file,Gaussian_datas_Compact):
    l = construct_list_of_attributes_Compact()
    print(len(l))
    xyz = Gaussian_datas_Compact[:,:3]
    normals = np.zeros_like(xyz)
    labels = Gaussian_datas_Compact[:,6].reshape((-1,1))
    opacities = Gaussian_datas_Compact[:,7].reshape((-1,1))
    scale = Gaussian_datas[:,8:11]
    rotation = Gaussian_datas[:,11:15]
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes_Compact()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, labels, opacities, scale, rotation), axis=1)
    print(attributes.shape)
    print(elements.shape)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(file)

def save_txt_Compact(file_codebook, centers):
    np.savetxt(file_codebook,centers)
def Compact(file,Gaussian_datas,centers,labels):
    file_codebook = file + "codebook.txt"
    file_ply = file + 'point_cloud_vq_compact.ply'
    Gaussian_datas_Compact = np.empty(shape=(len(Gaussian_datas),15))
    for i in range(len(Gaussian_datas)):
        label = labels[i]
        Gaussian_data = Gaussian_datas[i]
        Gaussian_datas_Compact[i][:6] = Gaussian_data[:6]
        Gaussian_datas_Compact[i][6] = label
        Gaussian_datas_Compact[i][7:] = Gaussian_data[54:62]
    # print(Gaussian_datas_Compact.shape)
    # print(Gaussian_datas_Compact[0])
    # print(Gaussian_datas[0][54:62])
    save_txt_Compact(file_codebook,centers)
    save_ply_Compact(file_ply,Gaussian_datas_Compact)



if __name__ == "__main__":
    file_path = 'data/output/point_cloud/iteration_30000/'
    Gaussian_datas = get_Gaussiandata(file_path)
    print(Gaussian_datas[1])
    centers, labels = MeanShift(Gaussian_datas)
    print(centers.shape, labels.shape)
    VQ_Gaussian(file_path,Gaussian_datas,centers,labels)
    # Compact(file_path,Gaussian_datas,centers,labels)


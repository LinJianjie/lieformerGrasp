import sys

from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
import numpy as np
import torch
from torch_cluster_sampling import BkNN


def bdot(a, b):  # it has the form [b,N1,N2] and [b,N1,N2]
    # assert (a.dim() == b.dim())
    return (a * b).sum(dim=a.dim() - 1)


def computeBatchPairFeatures(ps, pt, ns, nt):
    dptps = pt - ps
    dist = dptps.norm(dim=3)
    ns_copy = ns.clone().detach()
    nt_copy = nt.clone().detach()
    angle1 = bdot(ns_copy, dptps) / dist
    phi = angle1
    angle2 = bdot(nt_copy, dptps) / dist
    phi2 = angle2
    v = dptps.cross(ns_copy)
    v_norm = v.norm(dim=3, keepdim=True)
    v = v / v_norm
    w = ns_copy.cross(v)

    alpha = bdot(v, nt_copy)
    theta = torch.atan2(bdot(w, nt_copy), bdot(ns_copy, nt_copy))
    alpha = alpha.unsqueeze(dim=3)
    phi = phi.unsqueeze(dim=3)
    phi2 = phi2.unsqueeze(dim=3)
    theta = theta.unsqueeze(dim=3)
    dist = dist.unsqueeze(dim=3)
    localPF = torch.cat([theta, alpha, phi, phi2, dist], dim=3)  # B*N1*N2*4
    return localPF


class PointFeature:
    def __init__(self, points, maxKnn, includeSelf=False, queryPoints=None):  # points[batch_size, N, dim]
        self.points = points
        self.queryPoints = queryPoints
        self.maxKnn = maxKnn
        self.includeSelf = includeSelf
        self.__batchNormals = None
        self.__queryNormals = None
        self.knn_indices = None
        self.kdTree = BkNN(points, self.maxKnn, includeSelf, queryPoints=queryPoints)

    @property
    def batchNormal(self):
        return self.__batchNormals

    @batchNormal.setter
    def batchNormal(self, normals):
        self.__batchNormals = normals  # normals [batch,N,dim]

    @property
    def queryNormals(self):
        return self.__queryNormals

    @queryNormals.setter
    def queryNormals(self, normals):
        self.__queryNormals = normals  # normals [batch,N,dim]

    # def computeNormals(self):
    #     self.__batchNormals = EN.BatchEstimateNormals(self.points, self.maxKnn, self.includeSelf).normal

    def computeBatchSPF(self, queryIndex):
        ps, pt, ns, nt = self.findNN(queryIndex)
        simplifiedPointFeature = computeBatchPairFeatures(ps, pt, ns, nt)  # B*Nq*maxKnn*4
        if self.queryPoints is None:
            simplifiedPointFeature = simplifiedPointFeature.view(self.points.shape[0], queryIndex.size()[0], -1)
        else:
            simplifiedPointFeature = simplifiedPointFeature.view(self.queryPoints.shape[0], queryIndex.size()[0], -1)
        return simplifiedPointFeature

    def findNN(self, queryIndex):
        if self.queryPoints is not None:
            ps = self.queryPoints[:, queryIndex, :]
            ns = self.__queryNormals[:, queryIndex, :]
        else:
            ps = self.points[:, queryIndex, :]  # B*Nq*dim
            ns = self.__batchNormals[:, queryIndex, :]  # B*Nq*dim

        pt, indices = self.kdTree.queryIndics(queryIndex)  # pt --> B*Nq*maxKnn*dim
        self.knn_indices = indices
        nt = torch.stack([self.__batchNormals[i, indices[i, :], :] for i in range(self.__batchNormals.shape[0])])
        ps = ps.unsqueeze(dim=2).repeat(1, 1, self.maxKnn, 1)  # B*Nq*maxKnn*dim
        ns = ns.unsqueeze(dim=2).repeat(1, 1, self.maxKnn, 1)  # B*Nq*maxKnn*dim
        return ps, pt, ns, nt

    @staticmethod
    def createFullConnectGraph(index_):
        ind = torch.arange(0, index_).unsqueeze(dim=0)
        x1 = ind.repeat(ind.shape[1], 1).transpose(0, 1).reshape(1, -1)
        y1 = ind.repeat(ind.shape[1], 1).reshape(1, -1)
        xy = torch.cat((x1, y1), 0)
        tt = torch.arange(0, xy.shape[1], ind.shape[1] + 1)
        tt2 = torch.arange(0, xy.shape[1])
        tt3 = np.delete(np.asarray(tt2), np.asarray(tt))
        verticesIndex = xy[:, tt3]
        return verticesIndex

    @staticmethod
    def computePF(points, normals):
        # points=[b,N,k,dim], normals is [b,N,k,dim]
        verticesIndex = PointFeature.createFullConnectGraph(points.shape[2])  # 2*N2 --> N2=k*k-1
        # print("vertices: size: ", verticesIndex.shape)
        ps = points[:, :, verticesIndex[0, :], :]
        pt = points[:, :, verticesIndex[1, :], :]
        ns = normals[:, :, verticesIndex[0, :], :]
        nt = normals[:, :, verticesIndex[1, :], :]
        fpf = computeBatchPairFeatures(ps, pt, ns, nt)  # B*N*N2*4
        fpd = fpf.view(points.shape[0], points.shape[1], points.shape[2], -1)
        return fpd

import torch
import math
from numpy import sin, cos
import numpy as np
from collections import deque



def asd():
    class_num = 100
    feature_num = 64

    labels = torch.arange(class_num)
    feature = torch.nn.Linear(class_num, feature_num)
    classifier = torch.nn.Linear(feature_num, class_num)
    factor = torch.eye(class_num, class_num)
    softmax = torch.nn.Softmax(dim=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [
            {"params": feature.parameters(), "lr":1e-3, "momentum":0.9},
            {"params": classifier.parameters(), "lr":1e-3, "momentum":0.9},
        ]
    )

    c = classifier.weight
    f = feature.weight
    print("classifier: ", c.shape)
    print("feature: ", f.shape)

    aa = 1

    f_ = None
    def animate(i):
        global f_
        if aa == 1:
            c_ = torch.nn.functional.normalize(c, p=2, dim=1) * math.sqrt(class_num)
            f_ = torch.nn.functional.normalize(f, p=2, dim=0) * math.sqrt(class_num)

            pred_ = torch.mm(c_, f_)
            loss = loss_fn(pred_, labels)
            feature.zero_grad()
            classifier.zero_grad()
            # print("feature mean: ", torch.mean(f_, dim=1))
            # print("(pi) theta mean: ", math.acos(torch.mean(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item()) * 180 / math.pi)
            # print("theta mean: ", torch.mean(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item())
            # print("theta std: ", torch.std(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item())
        elif aa == 2:
            pred = torch.mm(c , f)
            loss = loss_fn(pred, labels)
            print("classifier norm: ", torch.mean(torch.norm(c, p=2, dim=1)).item())
            print("feature norm: ", torch.mean(torch.norm(f, p=2, dim=0)).item())
        loss.backward()
        optimizer.step()

        print("loss: ", loss.item())
        print()


    for i in range(1000):
        animate(i)

    print(feature.weight.data.shape)
    fff =  torch.nn.functional.normalize(feature.weight.data, p=2, dim=0)
    torch.save(fff, "f{}-c{}_feature".format(feature_num, class_num))

if __name__ == "__main__":
    asd()

# Grassmanian Frame
# if __name__ == "__main__":
#     class_num = 3
#     feature_num = 2
#     G = np.random.randn(class_num, class_num)
#     G = G.T @ G
#     from tqdm import tqdm
#     for i in tqdm(range(1000)):
#         H = np.clip(G, a_min=-1, a_max=1)
#         u, s, vh = np.linalg.svd(H, hermitian=True, compute_uv=True)
#         asd = np.sum(np.clip(np.expand_dims(s, axis=1) - np.expand_dims(s, axis=0), a_min=0,  a_max = 999), axis=0)        
#         for i in range(class_num-1):
#             if asd[i] < class_num < asd[i+1]:
#                 gamma = (np.sum(s[:i+1]) - class_num) / (i+1)
#                 break
#         else:
#             gamma = (np.sum(s) - class_num) / class_num
#         G = u @ np.diag(np.clip(s - gamma, a_min=0, a_max = 999)) @ vh

#     print("Gram Matrix: ")
#     print(G)
#     D = np.diag(np.diag(G) ** (-1/2))
#     print(D @ G @ D)

#     G = torch.from_numpy(G)
#     feature = torch.nn.Linear(feature_num, class_num)
#     optimizer = torch.optim.SGD(
#         [
#             {"params": feature.parameters(), "lr":1e-2, "momentum":0.9},
#         ]
#     )

#     for i in tqdm(range(1000)):
#         loss = torch.norm(feature.weight @ feature.weight.T - G)
#         loss.backward()
#         optimizer.step()
#         feature.zero_grad()

#     frames = feature.weight
#     frames = torch.nn.functional.normalize(frames, dim=1)
#     print("Norm: ", torch.norm(frames, dim=1))    
#     print("Gram Matrix: ", frames @ frames.T)
#     print("Tight: ", frames.T @ frames)

#     print("Frames: ", frames)


# import torch
# from numpy import sin, cos
# import numpy as np
# from collections import deque


# if __name__ == "__main__":

#     class_num = 3
#     feature_num = 2

#     labels = torch.arange(class_num)
#     feature = torch.nn.Linear(class_num, feature_num)
#     classifier = torch.nn.Linear(feature_num, class_num)
#     softmax = torch.nn.Softmax(dim=1)
#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(
#         [
#             {"params": feature.parameters(), "lr":1, "momentum":0.9},
#             {"params": classifier.parameters(), "lr":1, "momentum":0.9},
#         ]
#     )

#     c = classifier.weight
#     f = feature.weight
#     print("classifier: ", c.shape)
#     print("feature: ", f.shape)

#     aa = 1
#     c_ = None
#     f_ = None
#     def animate(i):
#         global f_, c_
#         if aa == 1:
#             # c_ = torch.nn.functional.normalize(c, p=2, dim=1)
#             # f_ = torch.nn.functional.normalize(f, p=2, dim=0)
#             c_ = c
#             f_ = f
#             pred_ = torch.mm(c_, f_)
            
#             import math
#             # print("feature mean: ", torch.mean(f_, dim=1))
#             # print("(pi) theta mean: ", math.acos(torch.mean(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item()) * 180 / math.pi)
#             print("theta mean: ", torch.mean(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item())
#             print("theta std: ", torch.std(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()]).item())
#             # loss = torch.std(torch.mm(f_.T, f_)[~torch.eye(class_num).bool()])
#             loss = loss_fn(pred_, labels)
#             # print("CE loss", ce_loss.item())
#         elif aa == 2:
#             pred = torch.mm(c , f)
#             loss = loss_fn(pred, labels)
#             print("classifier norm: ", torch.mean(torch.norm(c, p=2, dim=1)).item())
#             print("feature norm: ", torch.mean(torch.norm(f, p=2, dim=0)).item())

#         feature.zero_grad()
#         classifier.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print("loss: ", loss.item())
#         print()


#     for i in range(1000):
#         animate(i)

#     # torch.save(f_, "f{}-c{}_feature".format(feature_num, class_num))

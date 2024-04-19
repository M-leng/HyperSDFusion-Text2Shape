import torch
import torch.nn as nn

class ZSplit(nn.Module):
    def __init__(self):
        """ init method """
        super().__init__()

        self.share_layer = nn.Sequential(nn.Conv3d(3,16,1,1),
                                         nn.BatchNorm3d(16),
                                         nn.SiLU(),
                                         )
        self.base_layer = nn.Sequential(nn.Conv3d(16,16,1,1),
                                        nn.BatchNorm3d(16),
                                        nn.SiLU(),
                                        )

        self.graph_layer_1 = nn.Sequential(nn.Conv3d(16,16,3,1,1),
                                           nn.BatchNorm3d(16),
                                           nn.SiLU(),
                                           )

        self.graph_layer_2 = nn.Sequential(nn.Conv3d(16, 16, 5, 1, 2),
                                           nn.BatchNorm3d(16),
                                           nn.SiLU(),
                                           )

        self.graph_layer_3 = nn.Sequential(nn.Conv3d(16, 16, 7, 1, 3),
                                           nn.BatchNorm3d(16),
                                           nn.SiLU(),
                                           )

        self.graph_trans = nn.Sequential(nn.Conv3d(16*3, 16,1,1),
                                         nn.BatchNorm3d(16),
                                         nn.SiLU(),
                                         )

    def forward(self, x):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)

        #feature split
        x_share = self.share_layer(x)
        x_base = self.base_layer(x_share)
        x_graph_1 = self.graph_layer_1(x_share)
        x_graph_2 = self.graph_layer_2(x_share)
        x_graph_3 = self.graph_layer_3(x_share)
        x_graph = torch.cat([x_graph_1, x_graph_2, x_graph_3], dim=1)
        x_graph = self.graph_trans(x_graph)

        return x_base, x_graph



class ZMerge(nn.Module):
    def __init__(self):
        """ init method """
        super().__init__()

        self.out_base = nn.Sequential(nn.Conv3d(16, 16,1,1),
                                         nn.BatchNorm3d(16),
                                         nn.SiLU(),
                                         )

        self.out_graph = nn.Sequential(nn.Conv3d(16, 16,1,1),
                                         nn.BatchNorm3d(16),
                                         nn.SiLU(),
                                         )

        self.merge_layer = nn.Sequential(nn.Conv3d(16*2, 3,1,1),
                                         nn.BatchNorm3d(3),
                                         nn.SiLU(),
                                         nn.Conv3d(3,3,1,1))

    def forward(self, out_base, out_graph):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)

        #feature merge
        out_base = self.out_base(out_base)
        out_graph = self.out_graph(out_graph)
        out = torch.cat([out_base, out_graph], dim=1)
        out = self.merge_layer(out)
        return out
